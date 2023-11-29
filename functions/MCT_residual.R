
library(InstaPrism)

mixture = read.delim('/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP2/BayesPrism/MCT_pbmc_EXP2_randomprop_nonoise_mixture.csv',sep = ',',row.names = 1)
ref = read.delim('/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP2/bp_results/MCT_pbmc_EXP2_2missing_randomprop_nonoise_InstaPrism_usedref.csv',sep = '\t')
# cell.state = read.delim('MCT_pbmc_EXP2_randomprop_nonoise_1missing_cellstate.csv',sep = ',') # check if there is different cell states for a cell type
map = list()
for(ct in colnames(ref)){
  map[[ct]]=ct
}


refPhi_cs = new('refPhi_cs',
             phi.cs = as.matrix(ref),
             map = map)


deconv_res = InstaPrism(input_type = 'refPhi_cs',bulk_Expr = mixture,refPhi_cs = refPhi_cs,n.core = 16)
deconv_res_updated = InstaPrism_update(deconv_res,bulk_Expr = mixture,cell.types.to.update = 'all',key = NA,n.core = 16)


true_frac = read.delim('/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP2/BayesPrism/MCT_pbmc_EXP2_3missing_randomprop_nonoise_prop.csv',sep = ',',check.names = T)
deconv_performance_plot(deconv_res@Post.ini.ct@theta,t(true_frac))



# 1st way to get residual: use scaler from updated results to reconstruct X (using initial ref and intial theta)
reconstruct_Xhat = function(InstaPrism_obj,
                            InstaPrism_updated_obj,
                            pseudo.min=1E-8){
  
  theta_initial = InstaPrism_obj@Post.ini.cs@theta
  scaler_updated = InstaPrism_updated_obj@scaler
  
  phi_initial = InstaPrism_obj@initial.reference@phi.cs
  norm_factor = rowSums(phi_initial)
  norm_factor = ifelse(norm_factor==0,pseudo.min,norm_factor)
  phi_rownormalized = sweep(phi_initial,1,norm_factor,'/') # gene * cell.states
  
  stopifnot(all.equal(rownames(theta_initial),colnames(phi_rownormalized)))
  stopifnot(all.equal(rownames(phi_rownormalized),rownames(scaler_updated)))
  
  z_cs_from_scaler <-function(index){
    pp = theta_initial[,index] 
    intermediate = sweep(phi_rownormalized,2,pp,'*')
    z = sweep(intermediate,1,scaler_updated[,index],'*') 
    return(rowSums(z))
  }
  
  N = ncol(scaler_updated)
  
  reconstructed_X = do.call(cbind,lapply(seq(1,N),z_cs_from_scaler))
  colnames(reconstructed_X) = colnames(scaler_updated)
  
  return(reconstructed_X)
}
# one can also play around with different combinations of X reconstruction, for example, using updated scaler, updated theta and initial reference


reconstructed_X = reconstruct_Xhat(deconv_res,deconv_res_updated)


# 2nd way to get residual: use average scaler value from initial result to reconstruct X
reconstruct_XhatV2 = function(InstaPrism_obj,
                            pseudo.min=1E-8){
  
  theta_initial = InstaPrism_obj@Post.ini.cs@theta
  scaler_initial = InstaPrism_obj@initial.scaler
  
  phi_initial = InstaPrism_obj@initial.reference@phi.cs
  norm_factor = rowSums(phi_initial)
  norm_factor = ifelse(norm_factor==0,pseudo.min,norm_factor)
  phi_rownormalized = sweep(phi_initial,1,norm_factor,'/') # gene * cell.states
  
  stopifnot(all.equal(rownames(theta_initial),colnames(phi_rownormalized)))
  stopifnot(all.equal(rownames(phi_rownormalized),rownames(scaler_initial)))
  
  # plot(colSums(mixture),colSums(scaler_initial)) 
  # note the lib-size of scaler_initial is dependent on lib-size of mixture
  # so we will perform normalization on the scaler
  scaler_normed = sweep(scaler_initial,2,colSums(scaler_initial),'/')
  scaler_rowMean = rowMeans(scaler_normed) 

  z_cs_from_scaler <-function(index){
    pp = theta_initial[,index] 
    intermediate = sweep(phi_rownormalized,2,pp,'*')
    z = sweep(intermediate,1,scaler_rowMean *  sum(scaler_initial[,index]),'*') 
    return(rowSums(z))
  }
  
  N = ncol(scaler_initial)
  
  reconstructed_X = do.call(cbind,lapply(seq(1,N),z_cs_from_scaler))
  colnames(reconstructed_X) = colnames(scaler_initial)
  
  return(reconstructed_X)
}

reconstructed_X = reconstruct_XhatV2(deconv_res)
plot(mixture[rownames(reconstructed_X),1],reconstructed_X[,1])
plot(mixture[rownames(reconstructed_X),2],reconstructed_X[,2])

residual = c()
for(i in 1:1000){
  residual = c(residual,mean(abs(mixture[rownames(reconstructed_X),i] - reconstructed_X[,i])))
}


# consider if there's missing cell type in the reference
refPhi_cs_missing_one = refPhi_cs
refPhi_cs_missing_one@phi.cs = refPhi_cs_missing_one@phi.cs[,-1]
refPhi_cs_missing_one@map = refPhi_cs_missing_one@map[colnames(refPhi_cs_missing_one@phi.cs)]

deconv_res_missing_one = InstaPrism(input_type = 'refPhi_cs',bulk_Expr = mixture,refPhi_cs = refPhi_cs_missing_one,n.core = 16)
deconv_res_updated_missing_one = InstaPrism_update(deconv_res_missing_one,bulk_Expr = mixture,cell.types.to.update = 'all',n.core = 16)

reconstructed_X_missing_one = reconstruct_XhatV2(deconv_res_missing_one)
plot(mixture[rownames(reconstructed_X_missing_one),1],reconstructed_X_missing_one[,1])
plot(mixture[rownames(reconstructed_X_missing_one),2],reconstructed_X_missing_one[,2])


residual_missing_one = c()
for(i in 1:1000){
  residual_missing_one = c(residual_missing_one,mean(abs(mixture[rownames(reconstructed_X_missing_one),i] - reconstructed_X_missing_one[,i])))
}

boxplot(residual,residual_missing_one,names = c('residual','residual \n with missing one ct'))


resid_all_missing_one = reconstructed_X_missing_one - mixture[rownames(reconstructed_X_missing_one),]
resid_all = reconstructed_X - mixture[rownames(reconstructed_X),]

write.table(resid_all_missing_one, '/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP2/bp_results/residuals_2missing_chikina', sep="\t", quote=F, row.names = TRUE, col.names = TRUE,)
write.table(resid_all, '/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP2/bp_results/residuals_chikina', sep="\t", quote=F, row.names = TRUE, col.names = TRUE,)
