# Creating a confusion matrix with cvms
# https://cran.r-project.org/web/packages/cvms/vignettes/Creating_a_confusion_matrix.html
library(cvms)

library(broom)
library(tibble)
library(here)

################################################
# The confusion matrix for the 'default' model #
################################################

# Read CSV file containing prediction results.
predictions = read.csv(here('data', 'confidences.csv'))

# Build confusion matrix.
conf_mat = confusion_matrix(
  targets = predictions$expected_label,
  predictions = predictions$predicted_label)

# Render confusion matrix plot.
plot_confusion_matrix(conf_mat$`Confusion Matrix`[[1]],
                      #add_normalized = FALSE,
                      #add_col_percentages = FALSE,
                      #add_row_percentages = FALSE,
                      palette = 'Greens') # Try YlGnBu or PuBuGn.