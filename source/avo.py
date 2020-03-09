# Data accessing
import os

# Math
import numpy as np
from scipy import stats

# Data management
import pandas as pd
from shutil import copyfile
import time

# SEGY files management
import segyio

# Visualization
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool
from holoviews import streams #(AVO Visualization)
import panel as pn
hv.extension('bokeh')

# MUltiproccessing 
from multiprocessing import Pool


def input_generator(gather_file, list_of_angles, gradient_path, intercept_path, rvalue_path, pvalue_path, stderr_path):
    
    """
    
    Makes a generator of list for the AVO_attributes_computation_2 arguments.

    The multiprocess function pool.map() works with *args (requires a list), therefore this function makes a 
    generator that contains a list of arguments for the AVO_attributes_computation_2 and iterates along 
    seismic lines within the SEG-Y files to provide the parameters for a single trace instead of having a for loop.
    
    Arguments
    ---------
    gather_file : str
        Path where the text or Merged SEG-Y file is located.

    list_of_angles : list
        Contains the angles of each partial angle stack to be used.

    gradient_path : str
        Path of the file where the Gradient (AVO attribute) will be stored.
    
    intercept_path : str
        Path of the file where the Interpcet (AVO attribute) will be stored.
        
    rvalue_path : str
        Path of the file where the Correlation Coeficient (AVO attribute's statistiscs) will be stored.
    
    pvalue_path : str
        Path of the file where Gradient = 0 hypothesis (AVO attribute's statistiscs) will be stored.
    
    stderr_path : str
        Path of the file where the Standard Deviation (AVO attribute's statistiscs) will be stored.
        
    Return
    ------
    generator
        A generator that iterates along the seismic lines within the merged SEG-Y to provide argumets to work
        with a single trace within this file.
    
    See also
    --------
        1) utils.merging_stacks
        2) utils.avo_storing_files
        3) AVO_attributes_computation_2
        4) avo_storeputation

    """
    index = 0
    with segyio.open(gather_file, "r") as gather:
        for iline_number in gather.ilines:
            for xline_number in gather.xlines:
                    yield [gather_file, list_of_angles, iline_number, xline_number, index, 
                           gradient_path, intercept_path, rvalue_path, pvalue_path, stderr_path]
                    index += 1               

def AVO_attributes_computation_2(generator_list):
    
    """

    Computes and stores the AVO attributes besides the statistics parameters related to them.

    This function computes the AVO attributes using Scipy's linear regression by each sample in a single trace. 
    Among the Scipy's function output the Intercept, Slope (Gradient) and some statistics related to the regression
    can be found. All the parameters mentioned before are stored in different SEG-Y files, so that each parameter
    computed has coordinates related to the Seismic Survey (utmx, utmy and depth).
    
    Argument
    ---------
    generator_list : generator
        A generator that iterates along the seismic lines within the merged SEG-Y to provide argumets to work
        with a single trace within this file.
    
    Return
    ------
    str
        Indication of which AVO attributes, related to a trace, has been stored successfully.

    References
    ----------
    Scipy.org. scipy.stats.linregress. Online Document: 
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.linregress.html
    
    Note
    ----
        Because of Multiprocessing's Pool.map() function does not works well with for loops, this function which is
        controlled by Pool.map() uses a generator to provide its arguments.

    See also
    --------
        1) utils.merging_stacks
        2) input_generator
        3) avo_storeputation

    """

    # Parameters to use in the functions
    gather_file = generator_list[0]
    list_of_angles = generator_list[1]
    iline_number = generator_list[2]
    xline_number = generator_list[3]
    index = generator_list[4]
    gradient_file = generator_list[5]
    intercept_file = generator_list[6]
    rvalue_file = generator_list[7]
    pvalue_file = generator_list[8]
    stderr_file = generator_list[9]
    
    # Initializing the sin(angle) list
    sins = []
    
    # Opening the gather to: 
                            #1) Create an empty array to store the values within the gather file
                            #2) Make multiple arrays to be used by segyio's trace function
    with segyio.open(gather_file, "r") as gather:
        gradient = np.zeros([len(gather.samples), ], dtype = "float32")
        intercept = np.zeros([len(gather.samples), ], dtype = "float32")
        rvalue = np.zeros([len(gather.samples), ], dtype = "float32")#Correlation coefficient.
        pvalue = np.zeros([len(gather.samples), ], dtype = "float32")
        stderr = np.zeros([len(gather.samples), ], dtype = "float32") #Standard error of the estimated gradient.
        
        # For loot to append the values for sins and to store the amplitudes in an array(3,1501)
        for angle in list_of_angles:
            sins += [np.sin(np.radians(angle)) * np.sin(np.radians(angle))]
            
            # If the angle is the first, then initialize the amp array
            if angle == list_of_angles[0]:
                amp = gather.gather[iline_number, xline_number, angle].reshape(len(gather.samples), 1)
            
            # If it's not the first angle in the list, concatenate to an existing variable (amp)
            else:
                amp = np.concatenate((amp, 
                                      gather.gather[iline_number, xline_number, angle].reshape(len(gather.samples), 
        
                                                                                               1)),axis=1)
        # the amp is was reshaped to (3, 1501) in order to introduce the amplitudes of a slice in the
        # Scipy function: has been VECTORIZED!!
        
        # For loop to select and to introduce values to Scipy's
        for row in range(amp.shape[0]):
            gradient[row], intercept[row], rvalue[row], pvalue[row], stderr[row] = stats.linregress(sins, amp[row])
        
    # Storing the data: Opening and closing the segy's. This should improve the perfomance of the function
    with segyio.open(gradient_file, "r+") as gradient_segy:
        gradient_segy.trace[index] = gradient

    with segyio.open(intercept_file, "r+") as intercept_segy:
        intercept_segy.trace[index] = intercept

    with segyio.open(rvalue_file, "r+") as rvalue_segy:
        rvalue_segy.trace[index] = rvalue

    with segyio.open(pvalue_file, "r+") as pvalue_segy:
        pvalue_segy.trace[index] = pvalue

    with segyio.open(stderr_file, "r+") as stderr_segy:
            stderr_segy.trace[index] = stderr
    
    print(f"AVO attributes for trace number {index}, [{iline_number},{xline_number}], has been stored successfully")
    return

def avo_storeputation(gather_file, list_of_angles, gradient_file, intercept_file, rvalue_file, pvalue_file, stderr_file):
    
    """
    
    Gives celerity to the computation and storing of the AVO attributes by using the machine's cores instead of
    memory ram to go through the process.

    This function serves as a manager of the AVO_attributes_computation_2(generator_list) by employing the
    cores of a machine instead of the memory ram. The Multiprocessing Pool.map() function distributes a single task
    to every jobless core, thefore every core will solve the computation of one trace at the time. 
    
    Argument
    ---------
    gather_file : str
        Path where the text or Merged SEG-Y file is located.

    list_of_angles : list
        Contains the angles of each partial angle stack to be used.

    gradient_path : str
        Path of the file where the Gradient (AVO attribute) will be stored.
    
    intercept_path : str
        Path of the file where the Interpcet (AVO attribute) will be stored.
        
    rvalue_path : str
        Path of the file where the Correlation Coeficient (AVO attribute's statistiscs) will be stored.
    
    pvalue_path : str
        Path of the file where Gradient = 0 hypothesis (AVO attribute's statistiscs) will be stored.
    
    stderr_path : str
        Path of the file where the Standard Deviation (AVO attribute's statistiscs) will be stored.
        
    
    Return
    ------
    str
        Indication of which AVO attributes, related to a trace, has been stored successfully.

    References
    ----------
    Python.org. multiprocessing — Process-based parallelism. Online Document: 
    https://docs.python.org/3/library/multiprocessing.html
    
    
    Note
    ----
        Because of Multiprocessing's Pool.map() function does not works well with for loops, this function which is
        controlled by Pool.map() uses a generator to provide its arguments.

    See also
    --------
        1) utils.merging_stacks
        2) input_generator
        3) AVO_attributes_computation_2
    
    """
    input_args = input_generator(gather_file, list_of_angles, 
                                    gradient_file, intercept_file, rvalue_file, pvalue_file, stderr_file)
    
    if __name__ == "avo":
        p = Pool(maxtasksperchild = 1)
        p.map(AVO_attributes_computation_2, input_args, chunksize=1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

def attributes_organization(gradient_file, intercept_file, rvalue_file, pvalue_file, stderr_file, 
                            inline_range, crossline_range, time_window):
    
    """
    
    Gives celerity to the computation and storing of the AVO attributes by using the machine's cores instead of
    memory ram to go through the process.

    This function serves as a manager of the AVO_attributes_computation_2(generator_list) by employing the
    cores of a machine instead of the memory ram. The Multiprocessing Pool.map() function distributes a single task
    to every jobless core, thefore every core will solve the computation of one trace at the time. 
    
    Argument
    ---------
    gradient_path : str
        Path of the file where the Gradient values (AVO attribute) are stored.
    
    intercept_path : str
        Path of the file where the Interpcet values (AVO attribute) are stored.
        
    rvalue_path : str
        Path of the file where the Correlation Coeficient values (AVO attribute's statistiscs) are stored.
    
    pvalue_path : str
        Path of the file where Gradient = 0 hypothesis values (AVO attribute's statistiscs) are stored.
    
    stderr_path : str
        Path of the file where the Standard Deviation values (AVO attribute's statistiscs) are stored.
    
    inline_range : tuple
        Inline window to work with the data.
    
    crossline_range : tuple
        Crossline window to work with the data.
        
    time_range : tuple
        time window to work with the data.
        
    Return
    ------
    pando : (Pandas)DataFrame
        A matrix composed by the coordinates (either Seismic or Utm) of all the samples within the given window.
        The information has been distributed according to the following columns:
            - iline: inline coordinate.
            - xline: crossline coordinate.
            - utmx: horizontal axis of the Universal Transversal Mercator coordinate system.
            - utmy: vertical axis of the Universal Transversal Mercator coordinate system. 
            - time_slice: time of the sample [ms].
            - gradient: Gradient AVO attribute.
            - intercept: Intercept AVO attribute.
            - rvalue: Correlation Coeficient of the AVO attribute linear regression.
            - pvalue: Hypothesis for gradient = 0 of the AVO attribute linear regression.
            - sdeviation: Standard Deviation of the AVO attribute linear regression.
          
    See also
    --------
        1) AVO_attributes_computation_2
        2) crossplot
        3) avo_visualization
    
    """

    with segyio.open(gradient_file, "r") as g, segyio.open(intercept_file, "r") as f:
        with segyio.open(rvalue_file, "r") as r, segyio.open(pvalue_file, "r") as p, segyio.open(stderr_file, "r") as s:
            pando = pd.DataFrame([])
            for i in range(g.tracecount):
                pando = pd.concat([pando, pd.DataFrame({"inline":g.header[i][189],
                                                        "crossline": g.header[i][193],
                                                         "utmx":g.header[i][181], "utmy":g.header[i][185],
                                                         "time_slice":g.samples,
                                                         "gradient":g.trace[i], "intercept":f.trace[i],
                                                         "rvalue":r.trace[i], "pvalue":p.trace[i], 
                                                         "sdeviation":s.trace[i],
                                                         "scalar" : g.header[i][71]})],
                                       ignore_index = True, axis="rows")
                
            pando['utmx'] = np.where(pando['scalar'] == 0, pando['utmx'],
                                            np.where(pando['scalar'] > 0, pando['utmx'] * pando['scalar'], 
                                                                    pando['utmx'] / abs(pando['scalar'])))
            pando['utmy'] = np.where(pando['scalar'] == 0, pando['utmy'],
                                            np.where(pando['scalar'] > 0, pando['utmy'] * pando['scalar'], 
                                                                    pando['utmy'] / abs(pando['scalar'])))
    
    pando = pando.drop(["scalar"], axis=1)
    
    return pando[(pando["time_slice"] >= time_window[0]) &
                 (pando["time_slice"] <= time_window[1]) &
                 (pando["inline"] >= inline_range[0]) &
                 (pando["inline"] <= inline_range[1]) &
                 (pando["crossline"] >= crossline_range[0]) &
                 (pando["crossline"] <= crossline_range[1])].reset_index(drop=True)

def crossplot(x_column, y_column, dataframe, scale_select_value, alpha):
    
    """
    
    Plots interactive crossplots according to the selected columns using Holoview's bokeh backend.
    
    Argument
    ---------
    x_column : str
        Name of the column to be chosen as X axis. The value could be given manually
        or by Panel's selection widget inside the avo_visualization function of the avo module.


    y_column : str
        Name of the column to be chosen as Y axis. The value could be given manually
        or by Panel's selection widget inside the avo_visualization function of the avo module.
        
    dataframe : (Pandas)DataFrame
        A matrix composed by all the traces inside the selected window to work with.
    
    Return
    ------
    layout : Holoviews element [DynamicMap]
        A interactive Crossplot.
          
    See also
    --------
        1) attributes_organization
        2) avo_visualization

    """
    # Tools to select data
    opts.defaults(opts.Points(tools=['box_select', 'lasso_select']))
    
    # Font size for plots
    f_size = {'ticks': '10pt', 'title': '20pt', 'ylabel': '15px', 'xlabel': '15px'}
    
    # Scale for the crossplot
    levels = np.linspace(dataframe[scale_select_value].min(),
                         dataframe[scale_select_value].max(), 100,
                         endpoint = True).tolist()

    # Preparing the data's plot
    data = hv.Points(dataframe, [x_column, y_column], vdims = scale_select_value)
    data.opts(size = 3, width = 600, height = 600, padding = 0.01,
              title = f"{x_column} vs {y_column}",fontsize = f_size,
              active_tools = ["box_select"],
              toolbar='above',
              framewise = True, show_grid = True,
              color = scale_select_value, color_levels = levels, cmap = "fire", colorbar = True)
    
    # Axis of plot
    x_axis = hv.Curve([(0,dataframe[y_column].min()), (0,dataframe[y_column].max())])
    x_axis.opts(color = "black", line_width = 0.5)
    y_axis = hv.Curve([(dataframe[x_column].min(), 0), (dataframe[x_column].max(), 0)])
    y_axis.opts(color = "black", line_width = 0.5)
    
    # Declare points as source of selection stream
    selection = streams.Selection1D(source = data)
       
    # Write function that uses the selection indices to slice points and compute stats
    def selected_info(index):
        hc = dataframe.iloc[index]
        plot = hv.Scatter(hc, ["utmx","utmy"])
        plot.opts(color = "red", size = 5, width = 600, height = 600, padding = 0.1, 
                  title = f"Position of the selected traces",fontsize = f_size, 
                  xformatter = "%.1f", yformatter = "%.1f",
                  toolbar = 'above',
                  tools = [HoverTool(tooltips=[('Trace [I/X]', f"@inline/@crossline"),
                                     ('Time [ms]', f"@time_slice")])],
                  framewise = True, show_grid = True)
      
        return plot

    dynamic_map = hv.DynamicMap(selected_info, streams = [selection])
         
    # Combine points and DynamicMap
    layout = ((data * x_axis * y_axis) + dynamic_map).opts(merge_tools=False)
    
    # Line Display
    
    
    # DEACTIVATE SOME USELESS TOOLTIPS!!!
    return (layout)

def plotly(dframe, dataframe):
    
    """

        (Currently on development)
    Plots an interactive 3D figure of selected points in the crossplot function using Holoview's plotly backend.

    The function scan all the traces inside the given window and plots those that were selected by the crossplot
    function.

    Argument
    ---------
    dframe : (Pandas)DataFrame
        A matrix composed by the selected points in the crossplot function.
        
    dataframe : (Pandas)DataFrame
        A matrix composed by all the traces inside the selected window to work with.
    
    Return
    ------
    layout : Holoviews element [Scatter3D]
        A interactive 3D plot.

    Note
    ----
        3D visualization with plotly is a fact. But it has to be done manually because of panel objects
        are not accessible: whatever is used by panel, the output will be a panel's function. Moreover
        two renders are not compatible in one plot: Bokeh vs Plotly, therefore the application of this function
        will be manual as long as Luis doesn't find another way to overcome this issue.

        PD: Bokeh doesn't have 3D visualization tools (yet)

    See also
    --------
        1) attributes_organization
        2) crossplot

    """
    f_size = {'ticks':10, 'title':25, "ylabel":15, 'xlabel':15, 'zlabel':15,
                'legend':8, 'legend_title':13} 
    
    
    traces_with_hc = pd.DataFrame([])
    for iline in dframe["inline"]:
        for xline in dframe["crossline"]:
            traces_with_hc = pd.concat([traces_with_hc, 
                                        dataframe[(dataframe["inline"] == iline) &
                                                         (dataframe["crossline"] == xline)]], 
                                                axis = 0)
    # Plotly
    cube = hv.Scatter3D((traces_with_hc["inline"],
                         traces_with_hc["crossline"],
                         traces_with_hc["time_slice"])).opts(color = "black", size=1,
                                                             backend = "plotly")
    hc_position = hv.Scatter3D((dframe["inline"],
                                dframe["crossline"],
                                dframe["time_slice"])).opts(cmap = 'reds', color = 'z', size=5,
                                                             backend = "plotly")
    overlay = cube * hc_position
    return overlay.opts(title = "Selected traces: 3D Visualization",
                        fontsize = f_size,
                        width = 800, height = 800, padding = 0.3,
                        xlabel = "Inline", ylabel = "Crossline", zlabel = "Time slice [ms]",
                        show_grid = True, show_legend = True, invert_zaxis = True,
                        backend = "plotly")

def line_dataframe(button, merged_file_path, iline_number, xline_number, time_window, 
                   time_interval, list_of_angles):
    
    with segyio.open(merged_file_path, "r") as f:
    
        # Initializing the dataframe and angle
        amp_df = pd.DataFrame([])

        # Making the Y axis
        t_axis = time_axis(merged_file_path)

        #Interpolate to:
        new_time = np.arange(t_axis[0],t_axis[-1] + time_interval, time_interval)
        amp_df["time_axis"] = new_time
        amp_df["seismic line"] = button
        amp_df["iline"] = iline_number
        amp_df["xline"] = xline_number

        # Trace array
        for angle in list_of_angles:

            # Making two more series: Negative amplitude and positive amplitude
            amp_df[f"amplitude{angle}"] = f.gather[iline_number,xline_number, angle]
            amp_df[f"positive_amplitude{angle}"] = f.gather[iline_number,xline_number, angle]
            amp_df[f"negative_amplitude{angle}"] = f.gather[iline_number,xline_number, angle]

            # Separating the values of the amplitudes for Area element
            amp_df.loc[amp_df[f"positive_amplitude{angle}"] < 0 , f"positive_amplitude{angle}"] = 0
            amp_df.loc[amp_df[f"negative_amplitude{angle}"] > 0 , f"negative_amplitude{angle}"] = 0
           
    return (amp_df[(amp_df.time_axis >= time_window[0]) &
                          (amp_df.time_axis <= time_window[1])].round(4))

def line_visualization(button, merged_file_path, iline_number, xline_number, iline_range, xline_range, 
              time_window,time_interval, list_of_angles):
    
    with segyio.open(merged_stacks, "r") as f:
               
        if button == "Inline":
            iline_range = [iline_number]
            xline_range = np.arange(xline_range[0], xline_range[1]+1)
            
        else:
            iline_range = np.arange(iline_range[0], iline_range[1]+1)
            xline_range = [xline_number]
        
        # Computing scale factor for the X axis
        scale_f = scaling_fact(merged_file_path)
        s_factor = 0
        xticks = []
        
        for iline in iline_range:
            for xline in xline_range:
                amp_df = line_dataframe(button, merged_file_path, iline, xline, time_window,
                                        time_interval, list_of_angles)
                for angle in list_of_angles:
                                 
                    amp_df[f"s_amplitude{angle}"] = amp_df[f"amplitude{angle}"] + s_factor
                    amp_df[f"s_negative_amplitude{angle}"] = amp_df[f"negative_amplitude{angle}"] + s_factor
                    amp_df[f"s_positive_amplitude{angle}"] = amp_df[f"positive_amplitude{angle}"] + s_factor
                    
                    # Hover designation
                    hover_w= HoverTool(tooltips=[('Time', '@time_axis'),
                                                 ('Amplitude', f"@amplitude{angle}")])
                    # Plotting the wiggle
                    wiggle = hv.Curve(amp_df, ["time_axis", f"s_amplitude{angle}"], [f"amplitude{angle}"],
                                      label = "W")
                    wiggle.opts(color = "black", line_width = 2, tools = [hover_w])

                    # Making the area plot more comfortable
                    x = amp_df["time_axis"]
                    y = s_factor
                    y2 = amp_df[f"s_negative_amplitude{angle}"]
                    y3 = amp_df[f"s_positive_amplitude{angle}"]
                    
                    # Fill in between: Holoviews Element
                    negative = hv.Area((x, y, y2), vdims=['y', 'y2'], 
                                       label = "-").opts(color = "red", line_width = 0)
                    positive = hv.Area((x, y, y3), vdims=['y', 'y3'],
                                       label = "+").opts(color = "blue", line_width = 0)        
                    fill_in_between = negative * positive

                    # Overlying the colored areas +  the zero phase wavelet
                    if s_factor == 0:
                        wiggle_display = wiggle * fill_in_between
                    else:
                        wiggle_display = wiggle_display * wiggle * fill_in_between

                    # For next iteration
                    s_factor = s_factor + scale_f
            
        # xticks    
        x = np.arange(0, s_factor, scale_f * len(list_of_angles))
        if button == "Inline":
            for line in range(len(xline_range)):
                nlines = len(xline_range)
                xticks += [(x[line],xline_range[line])]
                line = iline_number
        else:
            for line in range(len(iline_range)):
                nlines = len(iline_range)
                xticks += [(x[line],iline_range[line])]
                line = xline_number
            
            
        wiggle_display.opts(height = 600, width = 500 + (10*nlines), padding = 0.01,
                            ylabel = f"{button} {line}", xlabel = "Time [ms]",
                            show_grid = True, xaxis = "top", invert_axes = True, invert_yaxis=True,
                            title = f"Position of the selected traces",
                            xticks = xticks,
                            fontsize = {'ticks': '10pt', 'title': '20pt', 'ylabel': '15px', 'xlabel': '15px'}, 
                            xformatter = "%.0f", yformatter = "%.0f",
                            toolbar = 'above', framewise = True,
                            legend_position = 'top')
        
    return wiggle_display

def avo_visualization(seismic_dataframe, merged_file_path,
                      gradient_file, intercept_file, rvalue_file, pvalue_file, stderr_file, 
                     list_of_angles):
    
    """

     Plots interactive crossplots according to the selected columns using Holoview's bokeh backend.
    
    Argument
    ---------
    gradient_path : str
        Path of the file where the Gradient values (AVO attribute) are stored.
    
    intercept_path : str
        Path of the file where the Interpcet values (AVO attribute) are stored.
        
    rvalue_path : str
        Path of the file where the Correlation Coeficient values (AVO attribute's statistiscs) are stored.
    
    pvalue_path : str
        Path of the file where Gradient = 0 hypothesis values (AVO attribute's statistiscs) are stored.
    
    stderr_path : str
        Path of the file where the Standard Deviation values (AVO attribute's statistiscs) are stored.
    
    Return
    ------
    NdLayout : Panel layout element [Row]
        A collection of DynamicMaps and Panel's widgets listed as controls:
            - Two track bars to modify size of the window to work with.
            - A track bar to select the depth of the window.
            - Two Select methods to choose the two axis of the crossplots.
          
    See also
    --------
        1) attributes_organization
        2) crossplot

    """

    df = pd.DataFrame(columns = ["inline","crossline","time_slice",
                                 "gradient","intercept","rvalue","pvalue","sdeviation"])
    # Window Selection
    inst = pn.widgets.StaticText(name = "Window to work with", value = "")
    
    iline_range = pn.widgets.IntRangeSlider(name = 'Inline range',
                                            start = int(seismic_dataframe["iline"].min()), 
                                            end = int(seismic_dataframe["iline"].max()), 
                                            value = (int(seismic_dataframe["iline"].min()), 
                                                     int(seismic_dataframe["iline"].min())+1), 
                                            step = 1)
    
    xline_range = pn.widgets.IntRangeSlider(name = 'Crossline range',
                                            start = int(seismic_dataframe["xline"].min()), 
                                            end = int(seismic_dataframe["xline"].max()), 
                                            value = (int(seismic_dataframe["xline"].min()), 
                                                     int(seismic_dataframe["xline"].min())+1), 
                                            step = 1)
    
    # Time slice selection
    slice_slider = pn.widgets.IntRangeSlider(name = 'Time slice [ms]',
                                             start = 0, 
                                             end = 6000, 
                                             value = (2750, 3000), 
                                             step = 250)
    # alpha slice selection
    alpha = pn.widgets.FloatSlider(name = 'Transparency slider',
                                 start = 0, 
                                 end = 1, 
                                 value = 1, 
                                 step = 0.1)
    
    # Crossplot parameters
    axis = pn.widgets.StaticText(name = "Crossplots", value = "")
    x_axis = pn.widgets.Select(name = "X axis", 
                              options = list(df.columns),
                              value = "intercept")
    y_axis = pn.widgets.Select(name = "Y axis", 
                              options = list(df.columns),
                              value = "gradient")
       
    # Scale selection
    select_scale = pn.widgets.Select(name = "Points Scale", 
                                    options = list(df.columns),
                                    value = "sdeviation")
    
    inst2 = pn.widgets.StaticText(name = "Seismic line", value = "")
    
    # Buttons
    line_button = pn.widgets.RadioButtonGroup(name='Radio Button Group', 
                                              options=['Inline', 'Crossline'], button_type='success')
    
    # Line entry
    iline_input = pn.widgets.TextInput(name = 'Inline number',
                                       value= str(iline_range.value[0]))
    xline_input = pn.widgets.TextInput(name = 'Crossline number',
                                       value= str(xline_range.value[0]))
    
    # display line
    checkbox = pn.widgets.Checkbox(name = "Display lines")

    
    # Decorator to mess up with the API
    @pn.depends(iline_range.param.value, xline_range.param.value,
                slice_slider.param.value,
                x_axis.param.value, y_axis.param.value,
                select_scale.param.value, alpha.param.value,
                line_button.param.value,
                iline_input.param.value, xline_input.param.value,
                checkbox.param.value)
    
    def avo_stuff(iline_range, xline_range, slice_slider,
                  x_axis, y_axis, select_scale, alpha, line_button, iline_input, xline_input, 
                  checkbox):
        
        attribute_dataframe = attributes_organization(gradient_file, intercept_file, 
                                               rvalue_file, pvalue_file, stderr_file, 
                                               iline_range, xline_range,
                                               slice_slider)
        # Crossplot stuff
        crossplots = crossplot(x_axis, y_axis, attribute_dataframe, select_scale, alpha)
        
        if checkbox:
            # Line visualization
            crossplots += line_visualization(line_button, merged_file_path, 
                                              int(iline_input), int(xline_input),
                                              iline_range, xline_range,slice_slider, 4,
                                              list_of_angles)

        return(crossplots).opts(merge_tools=False)

    
    w_width = 250
    w_height = 50
    row1 = pn.Row(inst, height = 30, width = w_width)
    row2 = pn.Row(iline_range, height = w_height, width = w_width)
    row3 = pn.Row(xline_range, height = w_height, width = w_width)
    row4 = pn.Row(slice_slider, height = w_height, width = w_width)
    row5 = pn.Row(x_axis, y_axis, height = w_height, width = w_width)
    row6 = pn.Column(select_scale, alpha, inst2, checkbox, line_button, width = w_width)
    row8 = pn.Row(iline_input, xline_input, height = w_height, width = w_width)
    
    widgets = pn.WidgetBox(f"## AVO visualization", row1,
                                      row2,
                                      row3,
                                      row4,
                                      axis,
                                      row5,
                                      row6,
                                      row8)
                       
    return pn.Row(widgets, avo_stuff).servable()


