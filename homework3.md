## Homework 3 PDE Convolution 
## Akash

### Problem 1 

In this part my goal was to time the for loop of the convolution computation and then vectorize it using numpy broadcasting to get the higher speed up. In short this task has been achieved as we can see the speed is increased form 24 sec to 0.054 sec.

Here for the **part b** we are using vectorization of the for loops as we did in the last homework in Poisson.ipynb for jacobi method. The equation that we implemented is given here : 
$ u_0(x,y) = C $

$u_{n+1}(x,y) = u_n(x,y) + u'_n(x,y) \delta t$

$u'_{n+1}(x,y) = u'_n(x,y) + (c^2 \Delta u ) \delta t$

so using the for loops we can run the program in 24 sec for 50 iterations. While if we implement the following program in vectorized form, 

```
U_ = u_init
Ut_= ut_init
lU_= np.zeros_like(ut_init)
start = time.time()
for k in range(0,50):
    lU_[2:-2,2:-2] = 1./8 *(U_[1:-3,2:-2] + U_[3:-1,2:-2]+ U_[2:-2,1:-3]+ U_[2:-2,3:-1] - 4 * U_[2:-2,2:-2])
    U_ = U_ + Ut_
    Ut_ = Ut_ + 1/4.* (lU_)
    
end = time.time() - start
print(" time it takes for the convolution using Vectorization is {0:.3f}".format(end))

```

It runs the code in 0.054 sec for 50 iterations. Which is significant improvement over the for loop code. so the speed up i sof the order of *450X*.


### Problem 2

In this problem our goal is to use hardware optimized libraries of pyhton to improve the speed of the given problem.

