library(MASS)
library(nlshrink)
library(here)
library(rstudioapi)
library(tools)
library(RcppCNPy)
library(reticulate)
np <- import("numpy")

source("ExampleMaker_in_R.R")

data_path = file_path_as_absolute(".")


d_List = seq(50,500,by=50)
NN = 50

##Figure 6
c = 2
ExampleNumber=2

AvgTimeList=c()

for (d in d_List){
  print(d)
  n = ceiling(d/c)
  
  TimeSum=0
  
  AllDataMatrices <- vector("list", NN)
  AllPopEVs <- vector("list", NN)
  
  list_of_estimatedEVs <- list()
  list_of_dimsChosen <- list()
  for (i in 1:NN) {
    
    out <- PopEV_Y_maker(d = d, n = n, ExampleNumber = ExampleNumber)
    PopEV = out$PopEV
    Y = out$Y
    
    AllDataMatrices[[i]] <- Y
    AllPopEVs[[i]] <- PopEV
    
    
    if (ExampleNumber==3){
      dims_chosen = out$dims_chosen
      list_of_dimsChosen <- c(list_of_dimsChosen,list(dims_chosen))
    }
    
    #Ledoit-Wolf method
    startTime <- Sys.time()
    Estimator_LedoitWolf = tau_estimate(t(Y), k = 0, method = "nlminb", control = list())
    endTime <- Sys.time()
    list_of_estimatedEVs <- c(list_of_estimatedEVs,list(Estimator_LedoitWolf))
    TimeSum = TimeSum+as.numeric(endTime-startTime, units = "secs")
  }
  AvgTimeList = c(AvgTimeList,TimeSum/NN)
  A <- do.call(cbind, list_of_estimatedEVs)
  filename <- paste0("Fig6_LedoitWolf_Estimators_d=", d, "_c=", c, "_Ex", ExampleNumber, ".npy")
  output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",filename,sep='/')
  npySave(output_file_path, A)
  if (ExampleNumber==3){
    D <- do.call(cbind, list_of_dimsChosen)
    filename <- paste0("Fig6_DimsChosen_d=", d, "_c=", c, "_Ex", ExampleNumber, ".npy")
    output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",filename,sep='/')
    npySave(output_file_path, D)
  }
  
  filename <- paste0("Fig6_DataMatrices_c=", c, "_Ex", ExampleNumber, "_d=", d, ".npy")
  output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",filename,sep='/')
  AllDataMatrices <- np$array(AllDataMatrices)
  np$save(output_file_path, AllDataMatrices)
  
  filename <- paste0("Fig6_PopEVs_c=", c, "_Ex", ExampleNumber, "_d=", d, ".npy")
  output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",filename,sep='/')
  AllPopEVs <- np$array(AllPopEVs)
  np$save(output_file_path, AllPopEVs)
  
}
filename <- paste0("Fig6_AvgTimes_LedoitWolf_c=", c, "_Ex", ExampleNumber, ".npy")
output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",filename,sep='/')
npySave(output_file_path, AvgTimeList)
output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data","Fig6_d_List.npy",sep='/')
npySave(output_file_path, d_List)
