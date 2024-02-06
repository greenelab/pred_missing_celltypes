#The following script opens csv files created in Python
#and turns them into the appropriate format for Bayes Prism.
#It is assumed the files already have a 1-start index.
# Import necessary libraries
library(InstaPrism)
library(BayesPrism)

data_type = "adp"

# Define the combinations of noise_type and proportion_type
combinations <- expand.grid(noise_type = c("noise", "nonoise"),
                            proportion_type = c("realistic", "random"))

# Define the values for num_missing
num_missing_values <- c(0,2)

# Loop through num_missing values
for (num_missing in num_missing_values) {
  # Loop through each combination
  for (i in 1:nrow(combinations)) {
    # Extract the current combination
    noise_type <- combinations$noise_type[i]
    proportion_type <- combinations$proportion_type[i]
      
      ###################
      #check if running in RStudio
      if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
        # Get the directory where the R script resides in RStudio
        script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
      } else {
        # If not in RStudio, use a fallback method (e.g., set it manually)
        script_dir <- ""
      }
      
      # Define the relative path based on the current combination
      relative_path <- file.path(script_dir, "..", "data", "EXP3/")
      
      # Combine the script directory and relative path to get the full path
      import_path <- paste0(relative_path, "BayesPrism/")
      export_path <- paste0(relative_path, "BP_results/")
      
      # Define filenames based on the current combination
      mixture_file <- paste0(import_path, "MCT_", data_type, "_EXP3_", proportion_type,"prop_", noise_type, "_mixture.csv")
      signal_file <- paste0(import_path, "MCT_", data_type, "_EXP3_", proportion_type,"prop_", noise_type, "_", num_missing, "missing_signal.csv")
      cell_state_file <- paste0(import_path , "MCT_", data_type, "_EXP3_", proportion_type,"prop_", noise_type,"_", num_missing, "missing_cellstate.csv")
      export_file_prop <- paste0(export_path, "MCT_", data_type, "_EXP3_", num_missing, "missing_", proportion_type, "prop_", noise_type, "_InstaPrism_results.csv")
      export_file_ref <- paste0(export_path, "MCT_", data_type, "_EXP3_", num_missing, "missing_", proportion_type, "prop_", noise_type, "_InstaPrism_usedref.csv")
      
      # read, inspect, mixture file as RDS
      mixture_file_rds <- mixture_file
      mixture_data <- read.csv(mixture_file, stringsAsFactors = FALSE, row.names = 1)
      cat("Mixture Data:\n")
      print(head(mixture_data[,1:3]))
      
      # read, inspect, signal files as RDS
      signal_files_rds <- signal_file
      signal_data <- read.csv(signal_file, stringsAsFactors = FALSE, row.names = 1)
      cat("Signal Data:\n")
      print(head(signal_data[,1:3]))
      
      # read, inspect, and save the cell state files as CSV
      cell_state <- read.csv(cell_state_file)
      cat("Cell State Data:\n")
      print(head(cell_state))
      
      ########################################################
      ###### Roughly following IntaPrism tutorial: https://humengying0907.github.io/InstaPrism_tutorial.html
      bulk_Expr <- mixture_data
      sc_Expr <- signal_data
      cell_type_labels <- t(cell_state[1])
      cell_state_labels <- t(cell_state[2])
      
      InstaPrism.res = InstaPrism(input_type = 'raw',sc_Expr = sc_Expr,bulk_Expr = bulk_Expr,
                                  cell.type.labels = cell_type_labels,cell.state.labels = cell_state_labels)
      
      ##Export files: proportions estimated and references used
      cell_frac = InstaPrism.res@Post.ini.cs@theta
      write.table(cell_frac, export_file_prop, sep="\t", quote=F,  row.names = TRUE, col.names = TRUE,)
      cell_ref = InstaPrism.res@initial.reference@phi.cs
      write.table(cell_ref, export_file_ref, sep="\t", quote=F, row.names = TRUE, col.names = TRUE,)
    
    # Printing the value of num_missing and combination to keep track
    cat(paste("Processing for num_missing =", num_missing, "with noise_type =", noise_type, "and proportion_type =", proportion_type), "\n")
  }
}
