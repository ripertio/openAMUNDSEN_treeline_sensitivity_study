
from cmath import inf


def get_r2_python(x_list, y_list):
    import math
    n = len(x_list)
    x_bar = sum(x_list)/n
    y_bar = sum(y_list)/n
    x_std = math.sqrt(sum([(xi-x_bar)**2 for xi in x_list])/(n-1))
    y_std = math.sqrt(sum([(yi-y_bar)**2 for yi in y_list])/(n-1))
    zx = [(xi-x_bar)/x_std for xi in x_list]
    zy = [(yi-y_bar)/y_std for yi in y_list]
    r = sum(zxi*zyi for zxi, zyi in zip(zx, zy))/(n-1)
    return r**2


def get_rmse(predictions, targets):
    import numpy as np
    return np.sqrt(((predictions - targets) ** 2).mean())

def get_nse(predictions, targets):
    import numpy as np
    return (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))


def compareObsandPred(obs,pred,outDir, prefix, abbildungsTitel, colsPred,colsObs):
#     if os.path.isdir(outDir):
#     print("OutDir exists.")
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    import scipy
    # from cmath import inf
    import pandas as pd
    import os
    import sklearn
    import numpy as np
    import math
    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    from sklearn.metrics import mean_absolute_error as get_mae


    #outStatistics = prefix + r"_result_stats.txt"
    outTableObs = prefix + r"_result_pred.csv"
    outTablePred = prefix + r"_result_obs.csv"

    outFigure = prefix + r"_result_obs_pred.jpg"



    #outStatistics = os.path.join(outDir,outStatistics)
    outObs = os.path.join(outDir,outTableObs)
    outPred = os.path.join(outDir,outTablePred)

    outFigure = os.path.join(outDir,outFigure)

    #Erzeuge DF
    obs_df = pd.read_csv(obs)
    pred_df = pd.read_csv(pred)

    #Reduzieren auf relvante Spalten
    pred_df = pd.DataFrame(data = pred_df, columns= colsPred)
    for index, row in obs_df.iterrows():
        #print(row['Date'])
        date = row[colsObs[0]].split(" ")
        dmy = date[0]
        dmy = dmy.split(".")
        day = dmy[0]
        month = dmy[1]
        year = dmy[2]
        time = date[1].split(":")
        hour = time[0]
        minute = time[1]

        datum = day+"."+month+"."+year+" " + hour+":"+minute

        obs_df.at[index,colsObs[0]] = datum


    for index, row in pred_df.iterrows():
        date = row[colsPred[0]].split(" ")
        ymd = date[0]
        ymd = ymd.split("-")
        year = ymd[0]
        month = ymd[1]
        day = ymd[2]
        time = date[1]
        time = time.split(":")
        hour = time[0]
        minute = time[1]
        datum = day+"."+month+"."+year+" " + hour+":"+minute
        pred_df.at[index,colsPred[0]] = datum

    #DF zuschneiden:
    obs_df = pd.DataFrame(data = obs_df, columns= colsObs)
    
    merged_df = obs_df.merge(pred_df, left_on="Date", right_on="time", how="inner")
    #print(merged_df)

    
    merged_df.dropna(subset = [colsObs[1]], inplace=True)

    x = merged_df[colsObs[1]].values
    y = merged_df[colsPred[1]].values

    r2 = get_r2_python(x, y) #Range: 0 -> 1
    r2 = round(r2,2)
    #print("r2:\t",r2)

    rmse = mean_squared_error(x, y, squared=False) #Range: –∞ -> 1
    rmse = round(rmse,2)
    #print("rmse:\t",rmse)

    # #Alternative:
    # rmse = get_rmse(y, x)
    # print("rmse:\t",rmse)

    nse = get_nse(y, x) #Range: –∞ -> 1
    nse = round(nse,2)
    #print("nse:\t",nse)

    mae = get_mae(x,y) # 0 (perfect fit) -> ∞
    mae = round(mae,2)
    #print("mae:\t",mae)
    # print("####")
    stats = "r2:"+ str(r2) + "    rmse:"+str(rmse) +"    nse:"+ str(nse) +"    mae:"+str(mae)
    # print(stats)
    obs_out = pd.DataFrame(data = merged_df, columns= colsObs)
    pred_out = pd.DataFrame(data = merged_df, columns= colsPred)

    obs_out.to_csv(outObs)
    pred_out.to_csv(outPred)

    startDate = len(merged_df["Date"])
    startDate = merged_df["Date"][round(startDate/10)]
    startDate = dt.datetime.strptime(startDate,'%d.%m.%Y %H:%M').date()


    plt.close('all')
    depth_obs = pd.read_csv(outObs, parse_dates=True, index_col=0, squeeze=True)
    df_sim = pd.read_csv(outPred, parse_dates=True, index_col=0)
    obs = pd.DataFrame(data=dict(obs=depth_obs[colsObs[1]]))
    sim = pd.DataFrame(data=dict(sim =df_sim[colsPred[1]]))
    dates = merged_df[colsObs[0]]
    x = [dt.datetime.strptime(d,'%d.%m.%Y %H:%M').date() for d in dates]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    # plot lines
    plt.plot(x, obs, ".", ms = 1, label = "obs")
    plt.plot(x, sim, label = "pred")
    plt.text(startDate,-1.5,stats)
    plt.gcf().autofmt_xdate()
    plt.suptitle(abbildungsTitel, fontsize=15)
    plt.legend()

    plt.savefig(outFigure,dpi = 900, bbox_inches="tight")

    plt.show()
   
def reclassSnowdepth(inFile,outFile):
    import os
    from collections import defaultdict
    import re
    if inFile.endswith(".txt") or inFile.endswith(".asc"):
        # print("reading:",inFile)
        # print("writing:",outFile)
        with open(inFile,"r") as fObj:
            content = fObj.readlines()
        skip = 0
        metadata = r""
        rasterdata = r""
        cellDict = defaultdict(list)
        newCellDict = defaultdict(list)
        for line in content:
            if skip < 5:
                #get metadata here
                metadata += line
            if skip >= 5:
                #reclassify here
                cells = re.split(r'(\s+)', line)
                #cells = line.split()
                for cell in cells:
                    cellDict[cell].append(cell)
                    try: #for numbers
                        float(cell)
                        if cell == "nan":
                            rasterdata += "nan" #no data
                            newCellDict["nan"].append(cell)
                        elif float(cell) < 0.1:
                            rasterdata += "3" #kein schnee
                            newCellDict["3"].append(cell)
                        elif float(cell) >= 0.1:
                            rasterdata += "5" #Schnee
                            newCellDict["5"].append(cell)     
                    except:
                        rasterdata += cell

                    #rasterdata += " "
                rasterdata.strip(" ")
        #         rasterdata += "\n"
            skip += 1

        #reclassified content
        newFileContent = metadata + rasterdata

        #write reclassified file
        with open(outFile,"w") as fObj:
            fObj.write(newFileContent)



def reclassSentinel(inFile,outFile):
    import os
    from collections import defaultdict
    import re
    import functions
    import numpy as np
    from osgeo import gdal
    
    if inFile.endswith(".tif"):
        # print("reading:",inFile)

        obs = gdal.Open(inFile)
        obsArray = np.array(obs.GetRasterBand(1).ReadAsArray())
        [rows, cols] = obsArray.shape
        arr_min = obsArray.min()
        arr_max = obsArray.max()
        arr_mean = int(obsArray.mean())
        y = obsArray.shape[0]
        x = obsArray.shape[1]
        reclassArray = np.empty(obsArray.shape,dtype = "int")
        row = 0
        for oR in obsArray:
            col = 0
            for oC in oR:
                try:
                    int(oC)
                    if oC == 0: #no Snow
                        reclassArray[row,col] = 7
                    elif oC == 100: #Snow
                        reclassArray[row,col] = 11
                    elif oC == 205: #Clouds
                        reclassArray[row,col] = 13
                    elif oC == 254: #NoData
                        reclassArray[row,col] = 0
                    elif oC == 255: #NoData
                        reclassArray[row,col] = 0
                except:
                    print("unecpected Value at row/col",obsArray.shape)
                col += 1
            row +=1

        # print("writing:",outFile)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(outFile, cols, rows, 1, gdal.GDT_UInt16)
        outdata.SetGeoTransform(obs.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(obs.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(reclassArray)
        outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    

def clipRaster(inFile,outFile,mask):
    # print(inFile,outFile)
    from osgeo import gdal
    import numpy as np
    import matplotlib.pyplot as plt


    ds = gdal.Open(inFile)
    # print("ds:",ds)
    dsArray = np.array(ds.GetRasterBand(1).ReadAsArray())
    plt.figure()
    plt.imshow(dsArray)
    plt.colorbar()
    #print("before:")
    #plt.show()
    y = dsArray.shape[0]
    x = dsArray.shape[1]

    #print(dsArray)

    # clip 
    # make sure your raster data and shapefile have the same projection!
    dsClip = gdal.Warp(outFile, 
                       ds, 
                       cutlineDSName = mask,
                       cropToCutline = True, 
                       dstNodata = np.nan)

    # visualize
    array = np.array(dsClip.GetRasterBand(1).ReadAsArray())
    plt.figure()
    plt.imshow(array)
    plt.colorbar()
    #print("after:")
    #plt.show()

    # close your datasets!
    ds = dsClip = dsRes = dsReprj = None

    
    

def multiplyPredWithObs(predF,obsF,outRaster):
    from osgeo import gdal
    import os
    import numpy as np
    # Open the pred file:
    pred = gdal.Open(predF)
    predArray = np.array(pred.GetRasterBand(1).ReadAsArray())

    # Open the obs file:
    obs = gdal.Open(obsF)
    obsArray = np.array(obs.GetRasterBand(1).ReadAsArray())

    [rows, cols] = obsArray.shape
    arr_min = obsArray.min()
    arr_max = obsArray.max()
    arr_mean = int(obsArray.mean())
    y = obsArray.shape[0]
    x = obsArray.shape[1]

    if predArray.shape != obsArray.shape:
        print("Achtung, nicht die selben ausschnitte")


    multiplyArray = np.empty(obsArray.shape,dtype = "int")
    #print(obsArray.shape, multiplyArray.shape)


    row = 0
    for pR in predArray:
        col = 0
        for pC in pR:
            multiplyArray[row,col] = obsArray[row,col] * predArray[row,col] 
            col += 1
        row +=1



    # print("writing:",outRaster)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outRaster, cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(obs.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(obs.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(multiplyArray)
    outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band=None
    ds=None
    return multiplyArray

def asc2tif(inFile,outFile):
    from osgeo import gdal, osr
    drv = gdal.GetDriverByName('GTiff')
    ds_in = gdal.Open(inFile)
    ds_out = drv.CreateCopy(outFile, ds_in)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(31254)
    ds_out.SetProjection(srs.ExportToWkt())
    ds_in = None
    ds_out = None
    



def ACC(TP,FP,TN,FN):
    return (TP + TN)/(TP+TN+FP+FN)

def BIAS(TP,FP,TN,FN):
    return (TP+FP)/(TP+FN)

def CSI(TP,FP,TN,FN):
    return (TP)/(TP+FN+FP)






def areaPerformance(inPrediction,inSentinel,workspace,mask):
    from matplotlib.colors import from_levels_and_colors
    import matplotlib.pyplot as plt
    #from pylab import *
    from osgeo import gdal
    import os
    import numpy as np
    #######################################################################################
    #######################################################################################
    #######################################################################################


    reclassedPrediction = os.path.join(workspace,"reclassed_prediction.txt")
    print("reclassify snowdepth...")
    reclassSnowdepth(inPrediction,reclassedPrediction)

    
    #######################################################################################
    #######################################################################################
    #######################################################################################

    predTif = os.path.join(workspace,"pred_tif.tif")

    print("asc2tif...")
    asc2tif(reclassedPrediction,predTif)
    
    #######################################################################################
    #######################################################################################
    #######################################################################################

    reclassSentinelF = os.path.join(workspace,"reclassed_sentinel.tif")
    print("reclassify sentinel...")
    reclassSentinel(inSentinel,reclassSentinelF)

    #######################################################################################
    #######################################################################################
    #######################################################################################

    clippedSentinel = os.path.join(workspace,"clipped_obs.tif")

    clippedPrediction = os.path.join(workspace,"clipped_pred.tif")

    #######################################################################################
    #######################################################################################
    #######################################################################################
    print("clip sentinel to mask...")
    clipRaster(reclassSentinelF,clippedSentinel,mask)

    print("clip prediction to mask...")
    clipRaster(predTif,clippedPrediction,mask)
    
    #######################################################################################
    #######################################################################################
    #######################################################################################

    multipliedRaster = os.path.join(workspace,"multipliedRaster.tif")

    #######################################################################################
    #######################################################################################
    #######################################################################################
    print("Multiply pred with obs...")
    multipliedArray = multiplyPredWithObs(clippedPrediction, clippedSentinel, multipliedRaster)
    
    #######################################################################################
    #######################################################################################
    #######################################################################################
    print("Get TP,FP,TN,FN...")
    (unique, counts) = np.unique(multipliedArray, return_counts=True)
    frequencies = np.asarray((unique, counts)).T

    #print(frequencies)
    
    # print("------------------------###------------------------")
    #######################################################################################
    #######################################################################################
    #######################################################################################

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for line in frequencies:
        line = list(line)
        if line[0] == 55:
            TP += line[1]
        if line[0] == 35:
            FP += line[1]
        if line[0] == 33:
            FN += line[1]
        if line[0] == 21:
            TN += line[1]

    # print("TP:",TP)
    # print("FP:",FP)
    # print("TN:",TN)
    # print("FN:",FN)
    
    #######################################################################################
    #######################################################################################
    #######################################################################################
    print("Calc Acc, Bias, CSI...")
    acc = round(ACC(TP,FP,TN,FN),3)
    bias = round(BIAS(TP,FP,TN,FN),3)
    csi = round(CSI(TP,FP,TN,FN),3)

    # print("ACC:\t",acc)
    # print("Bias:\t",bias)
    # print("CSI:\t",csi)

    #######################################################################################
    #######################################################################################
    #######################################################################################
    
    outStats = os.path.join(workspace,"stats.txt")
    txt2write = "ACC:"+str(acc)+";Bias:"+str(bias)+";CSI:"+str(csi)+"\n"+"TP:"+str(TP)+";TN:"+str(TN)+";FP:"+str(FP)+";FN:"+str(FN)
    
    with open(outStats,"w") as file:
        file.write(txt2write)
    
    
    #######################################################################################
    #######################################################################################
    #######################################################################################




#     outFigure = os.path.join(workspace,"figure.png")
#     abbildungsTitel = os.path.basename(inSentinel)[:10]
#     ds = gdal.Open(multipliedRaster)

#     dsArray = np.array(ds.GetRasterBand(1).ReadAsArray())
#     plt.figure()
#     # cmap = cm.get_cmap('PiYG', 7)
#     cmap, norm = from_levels_and_colors([10,25,34,40,60,70],['blue','yellow',"orange","green","gray"])
#     plt.imshow(dsArray, cmap=cmap, norm = norm)
#     plt.text(150,-50,"ACC: "+str(acc))
#     plt.text(150,-40,"CSI:  "+str(csi))
#     plt.text(150,-30,"BIAS: "+str(bias))
#     plt.text(150,-10,"TP = %s, FP = %s, FN = %s, TN = %s"%(TP,FP,FN,TN))
#     #%s. You are %s." % (name, age)
#     plt.xticks([])
#     plt.yticks([])

#     plt.suptitle(abbildungsTitel, fontsize=15)
#     plt.colorbar()
#     plt.savefig(outFigure,dpi = 900, bbox_inches="tight")
#     print("------------------------###------------------------")
    
    
def plotRowsForData(workspace_upper,days):
    import os
    import shutil
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.colors import from_levels_and_colors
    import matplotlib.image as mpimg
    import numpy as np

    days = days
    


    fig_folder = os.path.join(workspace_upper,"multipliedRaster")

    if not os.path.isdir(fig_folder):
        os.mkdir(fig_folder)
    statsList = []
    for upper_folder in os.listdir(workspace_upper):
        if upper_folder == "multipliedRaster": #für mehrfachausführung
            continue
        subfolder = os.path.join(workspace_upper,upper_folder)
        for folder in os.listdir(subfolder):
            fig_name = folder+".tif"
            folder = os.path.join(subfolder,folder)
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file.endswith("multipliedRaster.tif"):
                        source = os.path.join(folder,file)
                        destination = os.path.join(fig_folder,fig_name)
                        if not os.path.isfile(destination):
                            shutil.copy(source, destination)
                        else:
                            pass
                            #print("Achtung, file existiert bereits und wird übersprungen:\n"+fig_name)
                    if file.endswith("stats.txt"):
                        statsF = os.path.join(folder,file)
                        performance = ""
                        pVals = ""
                        with open(statsF,"r") as fObj:
                            stats = fObj.read()

                            stats = stats.split()
                            perf = stats[0].split(";")
                            for v in perf:
                                performance += v +" "
                            perfV = stats[1].split(";")
                            for v in perfV:
                                pVals += v +" "
                        statsList.append([fig_name,performance,pVals])
                
                  


    #print(statsList)
    ####################################################################

    
    def getImgData(fig):
        figP = os.path.join(fig_folder,fig)
        img = mpimg.imread(figP)
        #img.text(0,-10,"%s %s"%(fig[11:-4],fig[:10]))
        stat1 = ""
        stat2 = ""
        for stats in statsList:
            if stats[0] == fig:
                stat1 = stats[1]
                stat2 = stats[2]
        #print(fig)
        return[img,fig[:-4],fig[:10],stat1,stat2]
        #imgplot = plt.imshow(img)
        #plt.show()

    image_datas = list(map(getImgData,os.listdir(fig_folder)))
    ####################################################################

    modes = int(len(image_datas)/days)


    figure_folder = os.path.join(workspace_upper,"dateFig")

    if not os.path.isdir(figure_folder):
        os.mkdir(figure_folder)

    
    def createAndWriteImg(days,image_datas,modes):
        c= 0
        for x in range(days):
            outFileName = image_datas[c][1]
            outPlot = os.path.join(figure_folder,outFileName+".jpg")
            plt.rcParams["figure.figsize"] = [15, 15]
            plt.rcParams["figure.autolayout"] = True
            for y in range(modes):
                plt.subplot(1, modes, y+1)
                cmap, norm = from_levels_and_colors([0, 1, 22,34,36,56,80], ['white', 'blue',"orange","red","green","grey"])
                #nodata,TN,FN,FP,TP,Clouds
                plt.imshow(image_datas[c][0], cmap=cmap, norm=norm)
                #date =image_datas[c][2]
                method = image_datas[c][1]
    #             plt.text(0,-12,method,fontsize = 6)
    #             plt.text(0,-5,date,fontsize = 3)
                plt.text(0,-12,method,fontsize = 30)
                # plt.text(0,-5,date,fontsize = 20)
                plotPerf = str(image_datas[c][3])
                plotVals = str(image_datas[c][4])
    #             plt.text(80,-3,plotPerf,fontsize = 2)
    #             plt.text(80,-6,plotVals,fontsize = 2)
                plt.text(0,-3,plotPerf,fontsize = 24,weight="bold")
                plt.text(0,-7,plotVals,fontsize = 20)
                plt.xticks([])
                plt.yticks([])
                c+=1
            #print(outPlot)
            #plt.show()
            plt.savefig(outPlot,dpi = 600, bbox_inches="tight")
            plt.clf()

    createAndWriteImg(days,image_datas,modes)
    #print(image_datas)
    
    
    
    
    
###########################################################################################################################
########################################################################################################################
########################################################################################################################
##########################################################################################


# Function Definitions Performing Modeling with openAMUNDSEN
############################################################

def perform_modeling(a_yml_file):
    import openamundsen as oa

    config = oa.read_config(a_yml_file)  # read in configuration file
    model = oa.OpenAmundsen(config)  # create OpenAmundsen object and populate unspecified parameters with default values
    model.initialize()
    model.run()


    # Function Definitions Processing & Manipulating ONE VARIABLE in the yml files
###############################################################################  

def creating_new_yml_file (a_path, a_input_file, a_change_value):
    import yaml
    import os
    import regex
    import glob
    import json

    # manipulating yml file
    with open(a_input_file) as yml_file:
        dict_yml = yaml.safe_load(yml_file)
    
    max_depth = len(a_change_value[0])
    
    
    try:
        if type(a_change_value[1]) is list:
            outstring = ""
            keys = a_change_value[1][0].keys()
            for key in keys:
                print(key)
                
                outstring += key + "_" +str(a_change_value[1][0][key]) + "_"
                print(outstring)
            outstring.strip("_")
            print(outstring)
            yaml_suffix = "_".join(a_change_value[0])+"_"+outstring
            print(yaml_suffix)
        print(yaml_suffix)
    except:
        yaml_suffix = "_".join(a_change_value[0])+"_"+str(a_change_value[1])
    
    
    #looping through yml file
    for i in dict_yml:
        if i == "results_dir":
            #for x in yml_file[i]:
            dict_yml[i] = dict_yml[i] + "\\"+yaml_suffix
    
    if max_depth == 4:        
        dict_yml[a_change_value[0][0]][a_change_value[0][1]][a_change_value[0][2]][a_change_value[0][3]] = a_change_value[1]
    if max_depth == 3:        
        dict_yml[a_change_value[0][0]][a_change_value[0][1]][a_change_value[0][2]] = a_change_value[1]
    if max_depth == 2:        
        dict_yml[a_change_value[0][0]][a_change_value[0][1]] = a_change_value[1]

    #creating new yml file
    
    with open(f"{a_path}horlachtal_{yaml_suffix}.yml", "w") as new_yml_file:
        output = yaml.dump(dict_yml, new_yml_file)
    print(f"creating a new yaml file in {a_path} based on {a_input_file} and with the modified scf value of {a_change_value}")


def dataframe_erstellung(a_csv, a_name, a_column_name):
    import os
    import glob
    import regex
    import pandas as pd
    import numpy as np
    """Creating a pandas dataframe from the csv file, the name of the model, and the wanted column name
    All csv files must contain the column "time" and the colum {a_column_name} !!

    Args:
        a_csv (str): path to csv file
        a_name (str_): name of column in the created csv file
        a_column_name (str): name of column that is to be stored in the dataframe 

    Returns:
        _type_: _description_
    """
    b_name = a_name
    a_name = pd.read_csv(a_csv, parse_dates=["time"], dayfirst=True)
    if a_column_name in a_name.columns:
        a_name = a_name[["time",a_column_name]]
        a_name = a_name.rename(columns= {a_column_name: f"{a_column_name} {b_name}"})
    else:
        a_name = a_name["time"]
    #print(f"DEBUG: {a_column_name}")
    return a_name

def creat_dataframes_from_all_csv_files(a_path, a_name_of_column):
    import os
    import glob
    import regex
    import pandas as pd
    """Creates a dataframe from all csv files in an respective path. All files must contain

    Args:
        a_path (_type_): _description_
        a_name_of_column (_type_): _description_
    Returns:
        dict_: a dictionary containing all the csv files as dataframe {name: dataframe}
    """
    
    df_dic = {} #list of all panda dataframes
    for csv_files in glob.glob(os.path.join(a_path+"*", "*")):
        csv = regex.search(r".csv$", csv_files)
        if csv:
            df_name = regex.sub(rf"{a_path}", "", csv_files)
            df_name = regex.sub(r".csv", "", df_name)
            df_name = regex.sub(r"/", "_", df_name)
            #print(csv_files, type(df_name))
            df_dic[df_name] = dataframe_erstellung(csv_files, df_name, a_name_of_column)
    print(f"\nCreating a dataframe from all csv files in the path {a_path} with the criteria {a_name_of_column}")
    return df_dic

def create_combined_dataframe(a_dict_of_df, a_point_name):
    import os
    import glob
    import regex
    import pandas as pd
    """Creating combined dataframe from each dataframe that is created and stored in the dictionary a_dict_of_df.
    A_point_name is the title of the point/station that is to be validated 

    Args:
        a_dict_of_df (dic): _description_
        a_point_name (str): _description_

    Returns:
        type: (the combined dataframe, the name of the column of the observd values, a list of the modeld values)
    """
    print(f"Combining all the existing dataframes of the point {a_point_name} in one Dataframe ")
    df_lst_name = list() # list of names of pandas data frame
    df_combined = None
    for key,val in a_dict_of_df.items():
        df_lst_name.append(key)
    for i in df_lst_name:
        if df_combined is None:
            df_combined = a_dict_of_df[i].copy()
        else:
            df_combined = pd.merge(df_combined, a_dict_of_df[i], how="inner", on="time")
            df_combined.T.drop_duplicates().T
    
    column_names = []        
    for i in df_combined.columns.values: 
        if f"point_{a_point_name}" in i:

            column_names.append(i)
        if f"observed_{a_point_name}" in i:
            observed_name = i
    return (df_combined, observed_name, column_names)
    
    
# Function Definitions Validation
##########################


def R2_determination(df_col_obs, df_col_model):
    import os
    import glob
    import regex
    import pandas as pd
    import numpy as np
    """Calculating the R2 determination, requires that the oberseved values and modeled values are stored in a pandas. dataframe

    Args:
        df_col_obs ([pandas.core.series.Series]): Observed Values
        df_col_model ([pandas.core.series.Series]): Modeld Values

    Returns:
        [numpy.float]: R2 Determination for selected values]
    """
    observed_differ = df_col_obs.sub(df_col_obs.mean())
    modeld_differ = df_col_model.sub(df_col_model.mean())

    Denominator = observed_differ.mul(modeld_differ).sum()
    Devider = np.sqrt(observed_differ.pow(2).sum())*np.sqrt(modeld_differ.pow(2).sum())

    R2 = (Denominator/Devider)**2
    return R2

def NSE(df_col_obs, df_col_model):
    import os
    import glob
    import regex
    import pandas as pd
    import numpy as np
    """Calculates the Nash Sutcliffe Model Efficiency, requires the to arguments to be pandas.core.series.Series
    Args:
        df_col_obs ([pandas.core.series.Series]): Observed Values
        df_col_model ([pandas.core.series.Series]): Modeld Values

    Returns:
        [nunpy.float]: Nash Sutfliffe Efficiency Coefficient
    """
    Denominator = df_col_obs.sub(df_col_model).pow(2).sum()
    Devider = df_col_obs.sub(df_col_model.mean()).pow(2).sum()
    NSE = 1-(Denominator/Devider)
    return NSE

def PBIAS(df_col_obs, df_col_model):
    import os
    import glob
    import regex
    import pandas as pd
    import numpy as np
    """Calculates the Percept BIas (PBIAS) (obs-sim) requires the to arguments to be pandas.core.series.Series
    Args:
        df_col_obs ([pandas.core.series.Series]): Observed Values
        df_col_model ([pandas.core.series.Series]): Modeld Values

    Returns:
        [nunpy.float]: Percept Bias (PBIAS) subtracting obsved-simulated values
    """
    Denominator = df_col_obs.sub(df_col_model).sum()
    Devider = df_col_obs.sum()
    
    pbias = 100*Denominator/Devider
    return pbias

def IE(df_col_obs, df_col_model):
    import os
    import glob
    import regex
    import pandas as pd
    import numpy as np
    """ Calculates the Index of Agreement 
        requires the to arguments to be pandas.core.series.Series
    Args:
        df_col_obs ([pandas.core.series.Series]): Observed Values
        df_col_model ([pandas.core.series.Series]): Modeld Values

    Returns:
        [nunpy.float]:  Index of Agreement
    """
    Denominator = df_col_model.sub(df_col_obs).pow(2).sum()
    dev_abs = abs(df_col_model.sub(df_col_obs.mean()))+abs(df_col_obs.sub(df_col_obs.mean()))
    Devider = dev_abs.pow(2).sum()
    IE = 1-(Denominator/Devider)
    return IE

def MAE(df_col_obs, df_col_model):
    import os
    import glob
    import regex
    import pandas as pd
    import numpy as np
    """Calcualtes the Mean Absolut Error requires the to arguments to be pandas.core.series.Series
    Args:
        df_col_obs ([pandas.core.series.Series]): Observed Values
        df_col_model ([pandas.core.series.Series]): Modeld Values

    Returns:
        [nunpy.float]:  Mean Absolut Error
    """
    MAE = 1/len(df_col_obs)*df_col_obs.sub(df_col_model).abs().sum()
    return MAE

def RMSE(df_col_obs, df_col_model):
    import os
    import glob
    import regex
    import pandas as pd
    import numpy as np
    """Calculates the Root Mean Square Error requires the to arguments to be pandas.core.series.Series
    Args:
        df_col_obs ([pandas.core.series.Series]): Observed Values
        df_col_model ([pandas.core.series.Series]): Modeld Values

    Returns:
        [nunpy.float]: Root Mean Spuare Error (RMSE)a
    """
    RMSE = 1/len(df_col_obs)*df_col_obs.sub(df_col_model).pow(2).sum()
    RMSE = np.sqrt(RMSE)
    return RMSE

#######################
# Defining Function to print the criterias and ploting the observed and modeled values
#######################


def valided_point_dif_criteria (point_criteria, data_frame, name_df_col_obs, lst_df_col_model, a_plot_type):  
    import os
    import glob
    import regex
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    """prints the validation criteras: Determination R^2 Nash Sutcliffe Model Efficiency, Index of Agreement, Mean Absolut Error, Root Mean Square Error, Percept Bias and plots both values

    Args:
        point ([str]): Name of the respective Point
         df_col_obs ([pandas.core.series.Series]): Observed Values
        df_col_model ([pandas.core.series.Series]): Modeld Values
    """
    print(f"Punktvalidierung der Modelle {lst_df_col_model} für den Punkt {point_criteria} mit den {name_df_col_obs} Daten")

    df_col_obs = data_frame[name_df_col_obs]
    plt.subplots(figsize=(20,10))
    plt.scatter(data_frame["time"], df_col_obs, label=name_df_col_obs)
    for i in lst_df_col_model:
        print(f"\nPunktvalidierung für {i}")
        df_col_model = data_frame[i]
        R2 = R2_determination(df_col_obs, df_col_model)
        print("Determination of R2: ", R2)
        _NSE = NSE(df_col_obs, df_col_model)
        print("Neash Sutcliffe Efficiency: ", _NSE )
        _IE = IE(df_col_obs, df_col_model)
        print("Index of Agreement : ", _IE)
        _MAE = MAE(df_col_obs, df_col_model)
        print("Mean Absolut Error: ", _MAE)
        _RMSE = RMSE(df_col_obs, df_col_model)
        print("Root MEan Square Error: ", _RMSE)
        _PBIAS = PBIAS(df_col_obs, df_col_model)
        print("Percept Bias :", _PBIAS)
        if a_plot_type == "scatter":
            plt.scatter(data_frame["time"], df_col_model, label=i)
        elif a_plot_type == "line":
            plt.plot(data_frame["time"], df_col_model, label=i)
    
    plt.title(point_criteria)
    plt.legend()
    plt.show()
