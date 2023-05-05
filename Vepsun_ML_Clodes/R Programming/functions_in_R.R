#functions in R 

new.function<-function(x,y){
  res<-x+y
  print(res)
}

new.function(3,5)
new.function(x=3,y=8)


info.function<-function(x,y){
  res<-x*y
  print(res)
}

info.function(x=5,y=6)

data.function<-function(name,age=42){
  print(age)
}

data.function('Ramesh',)


#built in functions

x<--10
print(abs(x)) #abs will show the absolute value of the element

print(sqrt(25)) #sqrt will give the square root of the element

print(ceiling(5.6)) #ceiling will give the next whole number as result

print(ceiling(7.9))

print(ceiling(-9.8))

print(floor(5.6)) #floor will give the previous whole number as result

print(floor(6.9))

print(floor(7.1))

print(sin(5))

print(log(5))

print(exp(5))

x<-c(1.5,4.6,5.5,8.2,9.7)
print(trunc(x))

a1<-c(0:10,30,45,70,95)
print(a1)
s<-sum(a1)
print(s)
l<-min(a1)
print(l)
h<-max(a1)
print(h)

a<-"1,2,3,4,5,6,7"
substr(a,3,5)

s1<-"aashutosh"
print(toupper(s1)) #uppercase
s2<-"MISHRA"
print(tolower(s2)) #lowercase

























