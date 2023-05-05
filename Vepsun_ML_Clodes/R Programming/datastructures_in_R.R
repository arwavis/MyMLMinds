#Lists in R - list()
list1<-list("ram","shyam",c(1,2,3,4,5),TRUE,FALSE,52L,56L,23.56,92.81)
print(list1)

vec<-c(1,3,5,7,9)
char_vec<-c('India','USA','Bhutan','China')
list2<-list(vec,char_vec)
print(list2)

#naming of lists
list3<-list(c('ram','shyam','meera'),c(56,78,90),list('maths','chemistry','physics'))
names(list3)<-c('students','marks','subjects')
print(list3)

#accesing values from lists
#using index numbers 
print(list3[1])
print(list3[3])
#using names 
print(list3['marks'])
print(list3['subjects'])
#using $symbol
print(list3$students[[2]])
print(list3$marks)
print(list3$subjects[[3]])

#unlist() helps to convert a list into a vector
list4<-list(5:9)
list5<-list(14:18)
v1<-unlist(list4)
v2<-unlist(list5)
class(v1)
class(v2)

#merge two or more lists
mer<-list(list4,list5)
print(mer)

#Arrays in R - array()
v1<-c(1,4,5)
v2<-c(10,20,30,40,50,60,70)
v3<-array(c(v1,v2),dim=c(3,4,4)) #3 is the number of rows, 4 is the number of columns and 4 is the number of matrix that we want 
print(v3)

row_name<-c("row1","row2","row3")
col_name<-c("col1","col2","col3","col4")
mat_name<-c("mat1","mat2","mat3","mat4")
v3<-array(c(v1,v2),dim=c(3,4,4),dimnames = list(row_name,col_name,mat_name))

print(v3)

#accesing values in arrays
print(v3[3,2,1])#3 is 3rd row, 2 is 2nd column and 1 is the 1st matrix
print(v3[3,2,2])
print(v3[3,,2])
print(v3[,2,1])
print(v3[,,2])

#Matrix - matrix()

mat1<-matrix(c(2:13),nrow=4,ncol=3,byrow=FALSE)
print(mat1)
mat2<-matrix(c(2,5,4,8,7,9),nrow=2,ncol=3,byrow=TRUE)
print(mat2)

x<-matrix(c(5:16),nrow=4,byrow = TRUE)
y<-matrix(c(7:18),nrow=4,byrow=FALSE)
print(x)
print(y)
row_name<-c("row1","row2","row3","row4")
col_name<-c("col1","col2","col3")
mat<-matrix(c(7:18),nrow=4,byrow=TRUE,dimnames = list(row_name,col_name))
print(mat)
z<-matrix(c(5:15),nrow=4,byrow = TRUE)

#accesing values in matrix
print(mat[4,3])
print(mat[3,])
print(mat[,2])

mat[4,3]<-0 #element at 4th row and 3rd column will become 0
print(mat)

mat[mat==14]<-0 #whereever in the matrix we have the value 14, that gets changed to 0
print(mat)

mat[mat<10]<-0 #values which are less than 10 will be changed to 0 
print(mat)


















































