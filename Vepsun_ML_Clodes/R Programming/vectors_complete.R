#DataStructures in R

#Vector
#Lists
#Arrays
#Matrix
#DataFrames

#Vectors are of two types - atomic vector and lists
a<-c(3,4,5,6,7,1,9)
print(a)
b<--3:5
print(b)

#types of atomic vector

#numeric vector
numv<-c(12,6,7,8,9)
class(numv)
numvv<-c(12.4,5.7,23.45,145)
numvv<-as.numeric(numvv)
print(numvv)
class(numvv)

#integer vector
intv<-c(5,6,7,8,22,43,90)
intv<-as.integer(intv)
print(intv)
class(intv)
intvv<-c(45L,34L,78L,89L)
print(intvv)
class(intvv)

#charecter vector
charv<-c(2,3,4,5,6,9)
charv<-as.character(charv)
print(charv)
class(charv)
charvv<-c("ram","shyam","mohan","meera")
print(charvv)
class(charvv)

#accessing values from vectors
charv[4]
charvv[4]
intvv[3]
charvv[2:4]
charv[2:4]
numv[2:5]


