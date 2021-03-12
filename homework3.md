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

For this problem I looked into many different ways we we can take the full advantage of the Raspi 2 model 
B 1.1 .  Below are the ways I found interisting.

* GPU acceleration : Here I can take the advantage of GPU processing to do matrix calculation. Since GPU are made for it based on there multi cores. I tried to find a way to take advantage of this using GPGPU (general purpose GPU ). For that I looked for the OPENCL API. But found it hard implementing it. 
* There were many other discussion going on slack to implement it using different ways like using cython to write a code and then execute it. Cython gives us the speed up of 80X compare to normal pyhton. These could be a good way to do that but then it's more about the programming language then hardware.
* The last one that I ended up using is changing the Numpy version and find the one which gives us more speedup. I looked for different versions of Numpy from 1.18.5 to 1.15.4. Below is my results

                        Version 1.18.5   speed up : 0.045 to 0.047 s
                        Version 1.16.4   speed up : 0.037 to 0.045 s                all for 50 iteration of time step.
                        version 1.15.4   speed up : 0.037 to o.o43 s

here any version below 1.16.5 of Numpy is not supported by Scipy, but since we haven't used it in our problem we don't have to worry about that. But version 1.15.4 is ot supported by matplotlib, so we need to get rid of it. Hence the version I used for the numpy is 1.16.4.

The speed up is not significant as we wanted but it's something i guess for this small time step calculation. So I used the same code as in problem 1 but now with Numpy 1.16.4 .  It gave me speed up from 0.054 sec to 0.037 sec.