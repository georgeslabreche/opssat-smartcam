# Simple figures featuring different classes of thumbnails.

library(imager)
library(here)
library(container)

# Flag to indicate if plots should be displayed or written to file.
WRITE_PNG = FALSE

# DPI, or dots per inch.
# A measure of the resolution of a printed document or digital scan.
DPI = 72 # 72 is fine for small plots. Use 144 for larger plots.

# Plot size.
PLOT_HEIGHT_PX = 350
PLOT_WIDTH_PX = 1000

thumbnails_deque = Deque$new()
thumbnails_deque$add(c('Land.jpeg', 'Coast.jpeg', 'Sea.jpeg'))
thumbnails_deque$add(c('Earth.jpeg', 'Edge.jpeg', 'Bad.jpeg'))

for(index in 1:thumbnails_deque$size()){
  
  # Get thumbnails.
  thumbnails = thumbnails_deque$pop()
  
  # Write to file or display the plot.
  if(WRITE_PNG == TRUE){
    filename = paste0(paste(gsub('.jpeg', '', thumbnails), collapse=''), '.png')
    png(here('plots', filename), width=PLOT_WIDTH_PX, height=PLOT_HEIGHT_PX)
  }else{
    windows(height=PLOT_HEIGHT_PX/DPI, width=PLOT_WIDTH_PX/DPI)
  }
  
  # Remove title whitespace.
  par(mar=c(5,4,1,2)+0.1)
  
  # Place the 3 plots next to each other in a single row.
  layout(matrix(c(1, 2, 3), nrow=1, ncol=3, byrow=TRUE))
  
  # Get thumbnail images.
  img_earth = load.image(here('thumbnails', thumbnails[1]))
  img_edge = load.image(here('thumbnails', thumbnails[2]))
  img_bad = load.image(here('thumbnails', thumbnails[3]))
  
  # Plot thumbnails with subtitles.
  # Use the filename as the subtitle.
  
  # Earth.
  plot(img_earth, axes=FALSE)
  mtext(paste('(a)', strsplit(thumbnails[1], '\\.')[[1]][1]),
        side=1, line=1, cex=1.3, font=2)
  
  # Edge.
  plot(img_edge, axes=FALSE)
  mtext(paste('(b)', strsplit(thumbnails[2], '\\.')[[1]][1]),
        side=1, line=1, cex=1.3, font=2)
  
  # Bad.
  plot(img_bad, axes=FALSE)
  mtext(paste('(c)', strsplit(thumbnails[3], '\\.')[[1]][1]),
        side=1, line=1, cex=1.3, font=2)
  
  # Device off.
  if(WRITE_PNG == TRUE){
    dev.off()
  }

}