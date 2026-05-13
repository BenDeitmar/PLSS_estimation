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


n_List = seq(200,1000,by=100)

output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data","Fig12_n_List.npy",sep='/')
npySave(output_file_path, n_List)

NN = 50
d = 784

ExampleNumber=3

AvgTimeList=c()

for (n in n_List){
  print(n)
  
  TimeSum=0
  
  AllDataMatrices <- vector("list", NN)
  AllPopEVs <- vector("list", NN)
  
  list_of_estimatedEVs <- list()
  list_of_dimsChosen <- list()
  
  for (i in 1:NN) {
    repeat {
      ok <- tryCatch({
        
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
        
        TRUE
      }, error = function(e) {
        message("Error at i = ", i, ": ", conditionMessage(e))
        message("Retrying i = ", i)
        FALSE
      })
      
      if (ok) break
    }
    
    
  }
  AvgTimeList = c(AvgTimeList,TimeSum/NN)
  A <- do.call(cbind, list_of_estimatedEVs)
  filename <- paste0("Fig12_LedoitWolf_Estimators_n=", n, "_Ex", ExampleNumber, ".npy")
  output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",filename,sep='/')
  npySave(output_file_path, A)
  
  filename <- paste0("Fig12_DataMatrices_n=", n, "_Ex", ExampleNumber, ".npy")
  output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",filename,sep='/')
  AllDataMatrices <- np$array(AllDataMatrices)
  np$save(output_file_path, AllDataMatrices)
  
  filename <- paste0("Fig12_PopEVs_n=", n, "_Ex", ExampleNumber, ".npy")
  output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",filename,sep='/')
  AllPopEVs <- np$array(AllPopEVs)
  np$save(output_file_path, AllPopEVs)
  
  filename <- paste0("Fig12_DimsChosen_n=", n, "_Ex", ExampleNumber, ".npy")
  output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",filename,sep='/')
  DimsChosen <- np$array(list_of_dimsChosen)
  np$save(output_file_path, DimsChosen)
   
}
filename <- paste0("Fig12_AvgTimes_LedoitWolf_Ex", ExampleNumber, ".npy")
output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",filename,sep='/')
npySave(output_file_path, AvgTimeList)
