#EDA in R

#Pie Chart
#Histogram
#Bar Graph
#Line Graph
#Scatter Plot

#Pie chart - pie(x, labels, radius, main, col, clockwise)

#data for piechart
x<-c(12,45,72,56,23)
labels<-c("UK","JAPAN","USA","INDIA","GERMANY")
colour<-c("blue","orange","green","red","violet")

#creating a pie chart
pie(x,labels,main='Country pie chart',col=colour)

#creating a legend for the piechart - legend(x,y,labels,radius,col)
legend('topright',c("UK","JAPAN","USA","INDIA","GERMANY"),cex = 0.5, fill = colour)

#pie3D package allows us to create 3D piechart 

#Bar Graph - barplot(h, xlab, ylab, main, names.agr, col, border)

H1<-c(82,46,66,23,41)
barplot(H1)

H2<-c(12,35,54,31,41) #sales 
M2<-c("Jan","Feb","Mar","Apr","May")
barplot(H2,names.arg=M2,xlab="months",ylab="sales",col="yellow",main="Sales Bar Graph", border="red")
barplot(H2,names.arg=M2,xlab="months",ylab="sales",col="red",main="Sales Bar Graph", border="green")

#Group bar chart/Stacked bar chart

months<-c("Jan","Feb","Mar","Apr","May")
regions<-c("West","North","South")
values<-matrix(c(21,32,33,14,95,46,67,78,39,11,22,94,15,16,27), nrow = 3, ncol = 5, byrow = TRUE)
barplot(values,main="Revenue Bar Graph",names.arg=months, xlab="Months",ylab="Revenue",col=c("red","green","blue"))

legend("topright",regions,cex=0.5,fill=c("red","green","blue"))


#Histogram - hist()

v<-c(12,24,26,16,38,78,42,52,43,37,56,25)

hist(v,xlab="weight",ylab="frequency",col="green",border="red")

v<-c(12,24,26,16,38,78,42,52,43,37,56,25,66,61,64,62)
hist(v,xlab="weight",ylab="frequency",col="green",border="red")

hist(v,xlab="weight",ylab="frequency",col="orange",border="red",xlim = c(0,40),ylim = c(0,3))


#Line Graph - plot(v, type, col, xlab, ylab) 
#if type="l" - only line, if type="p" - only points, if type="o" - both line and points 

v<-c(18,22,28,7,31,52)
plot(v)
plot(v, type="o", col="red",xlab="months",ylab="rainfall")

#Line Graph with multiple lines
p<-c(2,3,4,3,15)
q<-c(3,6,5,4,2)
r<-c(8,3,6,7,9)

plot(p, type="o",col="green",xlab="months",ylab="rainfall")
lines(q,type="o",col="red")
lines(r,type="o",col="blue")

#Scatter Plot - plot(), ggplot()














































