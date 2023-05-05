#datatypes 
#numeric(default) = 45,-78, 3.14, 9.8,-11.34
#integer = 35L, -40L
#complex = 5+2i, 4-12i
#character = "India", "2346", "Sector15", "Gurgaon"
#logical = TRUE, FALSE 

a<-10
print(a)
class(a) #class() shows the datatype

b<-10L
class(b)


c<-10-25i
class(c)

d<-TRUE
class(d)

#converting one datatype to another 

#converting into numeric datatype
num1<-as.numeric(25L)
print(num1)

num2<-as.numeric(25-3i)
print(num2)

num3<-as.numeric("Aashutosh")
print(num3)

num4<-as.numeric("2535fksjk")
print(num4)

num5<-as.numeric("3457")
print(num5)

#converting into integer datatype
int1<-as.integer(25.45)
print(int1)

int2<-as.integer(45-35i)
print(int2)

int3<-as.integer("India")
print(int3)

int4<-as.integer("4500")
print(int4)

int5<-as.integer("4500India")
print(int5)

int6<-as.integer(TRUE)
print(int6)


int7<-as.integer(FALSE)
print(int7)
#converting into complex datatype
com1<-as.complex(35L)
print(com1)

com2<-as.complex("782")
print(com2)

com3<-as.complex("India3000")
print(com3)

#converting into logical datatype
log1<-as.logical(15)
print(log1)


log2<-as.logical(0)
print(log2)

log3<-as.logical(-5)
print(log3)

log4<-as.logical(15+5i)
print(log4)


log5<-as.logical("India3000")
print(log5)

log6<-as.logical("3000")
print(log6)

#converting into char datatype
char1<-as.character(55L)
print(char1)

char2<-as.character(30)
print(char2)

char3<-as.character(35-6i)
print(char3)

char4<-as.character(TRUE)
print(char4)

char5<-as.character(FALSE)
print(char5)






