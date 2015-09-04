---
layout: post
title: "A Secant Version of the L-M Method"
date: 2015-08-22 15:45
comments: true
categories: 
---

在优化的时候，基于梯度的的一系列算法是最常用的，但是还有一些情况下，我们的目标函数的梯度无法计算，这时候就需要用到 Secant version的一些算法了，比如 Secant version 的 L-M 算法。

函数的 Jacobian 矩阵计算方式如下：

$$
\mathbf{J} (x) = \left[ \cfrac{\partial f_i }{\partial x_j} \right]
$$

牛顿法有两个缺陷：

1）Jacibian 矩阵的计算过于复杂，需要计算 $$m*n$$ 次；同时在函数形式未知的情况下，需要很多的努力来估计 Jacobian 矩阵。

2）牛顿法在实践中，需要一个非常好的初始化点来收敛，而这个是不太容易满足的。

因此为了解决这个问题，出现了一系列的方法来减少对 Jacobian 矩阵的计算次数，可以做到只在初始的时候计算一次。

$$
\cfrac{\partial f_i }{\partial x_j} (\mathbf{x}) \approx \cfrac{f_i( \mathbf{x}+ \delta \mathbf{e}_j) - f_i(\mathbf{x})}{\delta} = b_{ij}
$$

$$\mathbf{e}_j$$ 是第 j 维为 0 的 unit vector. 在这种计算法方法下，每一轮我们需要 n+1 次 $$f(x) $$, 这是一种非常没有效率的做法。

Now consider

$$
\mathbf{f} (\mathbf{x}+\mathbf{h}) \approx \mathbf{l}(\mathbf{h}) = \mathbf{f} (\mathbf{x}) + \mathbf{J}(\mathbf{x}) \mathbf{h}
$$

由于我们无法确切知道 $$\mathbf{J}(\mathbf(x))$$，所以我们使用矩阵 $$\mathbf{B}$$ 来近似。即：

$$
\mathbf{f} (\mathbf{x}+\mathbf{h}) \approx \mathbf{l}(\mathbf{h}) = \mathbf{f} (\mathbf{x}) + \mathbf{B} \mathbf{h}
$$

通过泰勒展式展开可得

$$
f(x+h) = f(x) + Bh
$$

在这个式子中我们待求得是B，已知的是$$f(x+h), f(x), h$$, 所以这个有 n*n 个未知数和 n 个式子。因此 $$B$$ 是无法求解的。

为了解决这个问题，Broydon(1965) 提出一个新的条件，即 $$B_{new}$$ 和 $$B$$ 尽可能的相似。
所以我们解决的问题就变成了

$$
min ||B-B_{old}||

st. f(x+h) = f(x) + Bh
$$

这个问题有两种解释：

1、 在满足切线方程($$ f(x+h) - f(x) =  Bh $$, 这也是题目中 secant version 的由来)的前提之下，最小化 $$B$$ 和 $$B$$ 之间的  Frobenius norm diff.

2、第二种解释就是在 $$h$$ 所在的 $$n$$ 维空间中，有n个方向，和 $$\|\|B_{old}\|\|$$  在对 $$ \|\|h\|\|$$ 进行变换的过程中，只能在 $$h$$ 所在的方向上不同，在其他的基方向上，表现都应该相同。所以在其他 $$n-1$$ 个方向 $$v$$ 上，都应该满足 $$Bv = B_{old}v$$, 这样就有了额外 $$n-1$$ 个方程。所以一共联立起 $$n^2$$ 个方程，即可求解。

不管是基于 norm 求解的解释，还是基于空间基向量的表示，我们对上述问题进行求解之后，都可以得到一个解即

$$
B=B_{old}+uh^T

h=x-x_{old}, u = \cfrac{1}{h^Th}(f(x)-f(x_old)-Bh)
$$

在我们可以得到新的B之后，我们就可以套用传统的 L-M 方法，或者其他的 quasi-Newton 法来解目标问题了。a

这里我们将这个计算方法套入 L-M 算法当中就得到了 Secant Version 的 L-M 算法。

算法伪代码如下：

![](http://farm6.staticflickr.com/5786/20167391413_d5b6320014_b.jpg)

     %SMARQUARDT  Secant version of Levenberg-Marquardt's method for least   
     % Version 10.11.08.  hbn(a)imm.dtu.dk
     
     % Check parameters and function call
     f = NaN;  ng = NaN;  perf = [];  B = [];
     info = zeros(1,7);
     if  nargin < 2,  stop = -1;
     else
       [stop x n] = checkx(x0);   
       if  ~stop
     [stop f r] = checkrJ(fun,x0,varargin{:});  info(6) = 1;
     if  ~stop
       %  Finish initialization
       if  nargin < 3,  opts = []; end
       opts  = checkopts('smarquardt', opts);  % use default options where required
       tau = opts(1);tolg = opts(2);  tolx = opts(3);  relstep = opts(5);  
       if  opts(4) > 0,  maxeval = opts(4); else,  maxeval = 100 + 10*n; end
       % Jacobian
       if  nargin > 3  % B0 is given
     sB = size(B0);  m = length(r);
     if  sum(sB) == 0  % placeholder
       [stop B] = Dapprox(fun,x,relstep,r,varargin{:});  
       info(6) = info(6) + n;
     elseif  any(sB ~= [m n]),  stop = -4;
     else,  B = B0;   end
       else
     [stop B] = Dapprox(fun,x,relstep,r,varargin{:});  
     info(6) = info(6) + n;
       end
       % Check gradient and J'*J   
       if  ~stop
     g = B'*r;   ng = norm(g,inf);  A = B'*B;
     if  isinf(ng) | isinf(norm(A(:),inf)),  stop = -5; end 
       end
     end
       end
     end
     if  stop
       X = x0;  info([1:5 7]) = [f ng  0  tau  0 stop];
       return
     end
     
     % Finish initialization
     mu = tau * max(diag(A));
     Trace = nargout > 2;
     if  Trace
       o = ones(1, maxeval);  
       X = x * o;  perf = [f; ng; mu] * o;
     end 
     
     % Iterate
     k = 1;   nu = 2;   nh = 0;
     ng0 = ng;
     ku = 0;   % direction of last update
     kit = 0;  % no. of iteration steps
     
     while  ~stop
       if  ng <= opts(2),  stop = 1; 
       else 
     [h mu] = geth(A,g,mu);
     nh = norm(h);   nx = tolx + norm(x);
     if  nh <= tolx*nx,  stop = 2; end 
       end 
       if  ~stop
     xnew = x + h;h = xnew - x;  
     [stop fn rn] = checkrJ(fun,xnew,varargin{:});  info(6) = info(6)+1;
     if  ~stop
       % Update  B
       ku = mod(ku,n) + 1; 
       if  abs(h(ku)) < .8*norm(h)  % extra step
     xu = x;
     if  x(ku) == 0,  xu(ku) = opts(5)^2;
     else,xu(ku) = x(ku) + opts(5)*abs(x(ku)); end
     [stop fu ru] = checkrJ(fun,xu,varargin{:});  info(6) = info(6)+1;
     if  ~stop
       hu = xu - x;
       B = B + ((ru - r - B*hu)/norm(hu)^2) * hu';
     end
       end
       B = B + ((rn - r - B*h)/norm(h)^2) * h'; 
       k = k + 1;
       dL = (h'*(mu*h - g))/2;   
       if  length(rn) ~= length(r)
     df = f - fn;
       else  % more accurate
     df = ( (r - rn)' * (r + rn) )/2; 
       end 
       if  (dL > 0) & (df > 0)   % Update x and modify mu  
     kit = kit + 1;   
     x = xnew;   f = fn;  r = rn;
     mu = mu * max(1/3, 1 - (2*df/dL - 1)^3);   nu = 2;
     if  Trace
       X(:,kit+1) = x;   perf(:,kit+1) = [fn norm(B'*rn,inf) mu]'; end
       else  % Same  x, increase  mu
     mu = mu*nu;  nu = 2*nu; 
       end 
       if  info(5) > maxeval,  stop = 3; 
       else
     g = B'*r;  ng = norm(g,inf);  A = B'*B;
     if  isinf(ng) | isinf(norm(A(:),inf)),  stop = -5; end
       end
     end  
       end
     end
     %  Set return values
     if  Trace
       ii = 1 : kit+1;  X = X(:,ii);   
       perf = struct('f',perf(1,ii), 'ng',perf(2,ii), 'mu',perf(3,ii));
     else,  X = x;  end
     if  stop < 0,  tau = NaN;  else,  tau = mu/max(diag(A)); end
     info([1:5 7]) = [f  ng  nh  tau  kit stop];


当hessian矩阵为正定的时候，驻点为最小值；负定的时候为最大值；既有正的特征值又有负的特征值的时候，为鞍点