# Reproduced from: 
#   Szajnfarber, Z., Vrolijk, A., & Crusan, J. (2014). 
#   Exploring the interaction between open innovation methods and system complexity. 
#   In 4th international engineering systems symposium, Hoboken, New Jersey.
#   https://doi.org/10.1002/sys.21419

library(here)
library(RColorBrewer)

# Flag indicating whether or not to write plot to file or just display it in a new window.
WRITE_PNG = TRUE

# DPI, or dots per inch.
# A measure of the resolution of a printed document or digital scan.
DPI = 72 # 72 is fine for small plots. Use 144 for larger plots.

# Plot size.
PLOT_HEIGHT_PX = 350
PLOT_WIDTH_PX = 600

# Color palette.
pal = brewer.pal(n=5, name='Set2')

# Write to file or display the plot.
if(WRITE_PNG == TRUE){
  png(here('plots', 'crowd_quality.png'),
      height=PLOT_HEIGHT_PX, width=PLOT_WIDTH_PX, units='px', res=DPI)
}else{
  windows(height=PLOT_HEIGHT_PX/DPI, width=PLOT_WIDTH_PX/DPI)
}

# Remove title whitespace.
par(mar=c(3,3,1.2,2)+0.1)

# The mean of crowd distribution.
crowd_mean = 1

# The mean of expert distribution.
expert_mean = 1.25

# Create a sequence of numbers between 0 and 10 incrementing by 0.1.
expert_x = seq(0, 3, by=.01)

# Choose the mean as 2.5 and standard deviation as 0.5.
expert_y = dnorm(expert_x, mean=expert_mean, sd=0.1)

# Create a sequence of numbers between -10 and 10 incrementing by 0.1.
crowd_x = seq(0, 3, by=.01)

# Choose the mean as 2.5 and standard deviation as 0.5.
crowd_y = dnorm(crowd_x, mean=crowd_mean, sd=0.2)

# Plot expert Gaussian distribution.
plot(expert_x, expert_y, type='l', 
     xlab='',
     ylab='',
     xlim=c(0.35, 1.7),
     xaxt='n', yaxt='n', lwd=4, col=pal[2])

# X-axis label.
mtext('Solution quality', side=1, line=1)

# Y-axis label.
mtext('Solution frequency', side=2, line=1)

# Draw area in which the best crowd solution is better than the mean expert solution.
df = data.frame(crowd_x, crowd_y)
df = df[df$crowd_x >= expert_mean & df$crowd_x < 1.58,]

# This approach to filling the area under the line is taken from whuber:
# https://stackoverflow.com/a/29017246/4030804
# License: CC BY-SA 3.0
y = df$crowd_y
n = length(y)
x = df$crowd_x
s = smooth.spline(x, y, spar=0.5)
xy = predict(s, seq(min(x), max(x), by=0.01))   # Some vertices on the curve
m = length(xy$x)                         
x.poly = c(xy$x, xy$x[m], xy$x[1])              # Adjoin two x-coordinates
y.poly = c(xy$y, 0, 0)                          # ...and the corresponding y-coordinates
polygon(x.poly, y.poly, col='#f1b6da', border=NA)  # Show the polygon fill only

# Plot crowd Gaussian distribution.
lines(crowd_x, crowd_y,
      lwd=4, lty=2, col=pal[1])


# Plot mean of crowd distribution.
abline(v=crowd_mean, lty=4, lwd=2, col=pal[5])

# Plot mean of expert distribution.
abline(v=expert_mean, lty=6, lwd=2, col=pal[4])

# Plot right tail of crowd distribution.
abline(v=1.5, lty=3, lwd=2, col=pal[3])


# Legend
legend("topleft", inset=0.01, lwd=2,
       legend=c(
         'Crowd distribution',
         'Expert distribution',
         'Mean of crowd distribution', 
         'Mean of expert distribution',
         'Right tail of crowd distribution'),
       col=c(pal[1], pal[2], pal[5], pal[4], pal[3]),
       lty=c(2, 1, 4, 6, 3), 
       cex=1, box.lty=0, seg.len=4)

# Device off.
if(WRITE_PNG == TRUE){
  dev.off()
}