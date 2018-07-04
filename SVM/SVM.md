# SVM
## Prinsciple
原理介绍，以下视屏介绍的非常清除，推荐。
https://www.youtube.com/watch?v=6c0cvcxcZEA

推导过程我这里简单的复写以下，权当做笔记了。
svm是机器学习的方法之一。那么机器学习的思路就是从样本数据中学习，建立数学模型，然后应用数学模型，使之可以去解决类似的问题。
svm的思想是：通过训练，使得离超平面最近的点到超平面的距离最大。这个条件相当于：要找到距离超平面最近的点，使之到超平面的距离最大。用数学公式表示就是：

<a href="https://www.codecogs.com/eqnedit.php?latex=arg(\underset{W,b}{max}\left&space;\{&space;\underset{n}{min}(label\cdot(W^{T}x&plus;b))\cdot&space;\frac{1}{\left&space;\|&space;W&space;\right&space;\|}&space;\right&space;\})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?arg(\underset{W,b}{max}\left&space;\{&space;\underset{n}{min}(label\cdot(W^{T}x&plus;b))\cdot&space;\frac{1}{\left&space;\|&space;W&space;\right&space;\|}&space;\right&space;\})" title="arg(\underset{W,b}{max}\left \{ \underset{n}{min}(label\cdot(W^{T}x+b))\cdot \frac{1}{\left \| W \right \|} \right \})" /></a>

点到面的距离公式为：

<a href="https://www.codecogs.com/eqnedit.php?latex=d=\frac{\left&space;|&space;W^{T}x&plus;b&space;\right&space;|}{\left&space;\|&space;w&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d=\frac{\left&space;|&space;W^{T}x&plus;b&space;\right&space;|}{\left&space;\|&space;w&space;\right&space;\|}" title="d=\frac{\left | W^{T}x+b \right |}{\left \| w \right \|}" /></a>   (1)

其中平面可以标示为：

<a href="https://www.codecogs.com/eqnedit.php?latex=W^{T}x&plus;b=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{T}x&plus;b=0" title="W^{T}x+b=0" /></a>

样本分别标记为label(x)∈{-1, 1}。

接下来思考线性规划问题：在最理想的超平面两侧相同距离上存在着与理想超平面平行的平面，-1和1标签的样本开始位于这两个平面上，这两个平面可以分别用如下数学公式标示：

<a href="https://www.codecogs.com/eqnedit.php?latex=W^{T}x&plus;b=-1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{T}x&plus;b=-1" title="W^{T}x+b=-1" /></a>         (2)

<a href="https://www.codecogs.com/eqnedit.php?latex=W^{T}x&plus;b=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{T}x&plus;b=1" title="W^{T}x+b=1" /></a>           (3)


这两个平面之间的距离为：

<a href="https://www.codecogs.com/eqnedit.php?latex=d=\frac{2}{\left&space;\|&space;W&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d=\frac{2}{\left&space;\|&space;W&space;\right&space;\|}" title="d=\frac{2}{\left \| W \right \|}" /></a>  (4)

推导过程很简单，假设某侧有一个点，该点到两个平面的距离采用公式(1)得到，然后做差值就可以得到(4)。我们希望平面之间的距离d越大越好。d越大，也就是超平面到支持向量的距离越大，分类器泛化效果也就越好。

对于所有-1标签的样本而言，总有：

<a href="https://www.codecogs.com/eqnedit.php?latex=W^{T}x&plus;b\leqslant&space;-1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{T}x&plus;b\leqslant&space;-1" title="W^{T}x+b\leqslant -1" /></a>    (5)

而所有1标签的样本有：

<a href="https://www.codecogs.com/eqnedit.php?latex=W^{T}x&plus;b\geqslant&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{T}x&plus;b\geqslant&space;1" title="W^{T}x+b\geqslant 1" /></a>        (6)

公式（4）和（5）分别乘上标签值，可以得到：

<a href="https://www.codecogs.com/eqnedit.php?latex=label^{(i)}*(W^{T}x^{(i)}&plus;b)\geqslant&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?label^{(i)}*(W^{T}x^{(i)}&plus;b)\geqslant&space;1" title="label^{(i)}*(W^{T}x^{(i)}+b)\geqslant 1" /></a> (7)

这里就得到svm的约束条件了。可以看到，为什么标签值我们使用-1和1而不用其它值的原因了吧。

总结一下，对于svm，我们的目标是：

<a href="https://www.codecogs.com/eqnedit.php?latex=max:&space;d=\frac{2}{\left&space;\|&space;w&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?max:&space;d=\frac{2}{\left&space;\|&space;w&space;\right&space;\|}" title="max: d=\frac{2}{\left \| w \right \|}" /></a>     (8)

<a href="https://www.codecogs.com/eqnedit.php?latex=s.t.:&space;label^{(i)}*(W^{T}x^{(i)}&plus;b)\geqslant&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s.t.:&space;label^{(i)}*(W^{T}x^{(i)}&plus;b)\geqslant&space;1" title="s.t.: label^{(i)}*(W^{T}x^{(i)}+b)\geqslant 1" /></a>      (9)

对于公式8，求d最大值，其实也就是求||W||的最小值。考虑到通过偏导数来求极值，我们使用平方范数。也就是：

<a href="https://www.codecogs.com/eqnedit.php?latex=min:&space;d=\frac{\left&space;\|&space;W&space;\right&space;\|^{2}}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min:&space;d=\frac{\left&space;\|&space;W&space;\right&space;\|^{2}}{2}" title="min: d=\frac{\left \| W \right \|^{2}}{2}" /></a>     (10)

<a href="https://www.codecogs.com/eqnedit.php?latex=s.t.:&space;label^{(i)}*(W^{T}x^{(i)}&plus;b)\geqslant&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s.t.:&space;label^{(i)}*(W^{T}x^{(i)}&plus;b)\geqslant&space;1" title="s.t.: label^{(i)}*(W^{T}x^{(i)}+b)\geqslant 1" /></a>      (11)

为了求解带约束的求极值问题，可以通过引入拉格朗日乘子，将不等式约束条件化为相同的求极值问题，这样公式10和11的问题就可以转化为以下问题：

<a href="https://www.codecogs.com/eqnedit.php?latex=min:&space;L(w,b,\alpha&space;)=\frac{\left&space;\|&space;w&space;\right&space;\|^{2}}{2}-\sum_{i=1}^{n}\alpha_{i}(label^{(i)}*(W^{T}x^{(i)}&plus;b)-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min:&space;L(w,b,\alpha&space;)=\frac{\left&space;\|&space;w&space;\right&space;\|^{2}}{2}-\sum_{i=1}^{n}\alpha_{i}(label^{(i)}*(W^{T}x^{(i)}&plus;b)-1)" title="min: L(w,b,\alpha )=\frac{\left \| w \right \|^{2}}{2}-\sum_{i=1}^{n}\alpha_{i}(label^{(i)}*(W^{T}x^{(i)}+b)-1)" /></a>    （12）

对于12，可以通过求导数等于0来求极值。对W和b求偏导数，可以得到：

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L}{\partial&space;W}=W-\sum_{i=1}^{n}\alpha_{i}label^{(i)}*x^{(i)}=0&space;=>&space;W=\sum_{i=1}^{n}\alpha_{i}label^{(i)}*x^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;W}=W-\sum_{i=1}^{n}\alpha_{i}label^{(i)}*x^{(i)}=0&space;=>&space;W=\sum_{i=1}^{n}\alpha_{i}label^{(i)}*x^{(i)}" title="\frac{\partial L}{\partial W}=W-\sum_{i=1}^{n}\alpha_{i}label^{(i)}*x^{(i)}=0 => W=\sum_{i=1}^{n}\alpha_{i}label^{(i)}*x^{(i)}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L}{\partial&space;b}=\sum_{i=1}^{n}\alpha_{i}label^{(i)}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;b}=\sum_{i=1}^{n}\alpha_{i}label^{(i)}=0" title="\frac{\partial L}{\partial b}=\sum_{i=1}^{n}\alpha_{i}label^{(i)}=0" /></a>

将这两个结果代入公式12：

<a href="https://www.codecogs.com/eqnedit.php?latex=min:&space;L(\alpha&space;)=\frac{1}{2}\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}label^{(i)}label^{(j)}x^{(i)T}x^{j}-\sum_{i=1}^{n}\alpha_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min:&space;L(\alpha&space;)=\frac{1}{2}\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}label^{(i)}label^{(j)}x^{(i)T}x^{j}-\sum_{i=1}^{n}\alpha_{i}" title="min: L(\alpha )=\frac{1}{2}\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}label^{(i)}label^{(j)}x^{(i)T}x^{j}-\sum_{i=1}^{n}\alpha_{i}" /></a>

上式等效于：

<a href="https://www.codecogs.com/eqnedit.php?latex=max:&space;L(\alpha&space;)=\sum_{i=1}^{n}\alpha_{i}&space;-&space;\frac{1}{2}\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}label^{(i)}label^{(j)}x^{(i)T}x^{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?max:&space;L(\alpha&space;)=\sum_{i=1}^{n}\alpha_{i}&space;-&space;\frac{1}{2}\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}label^{(i)}label^{(j)}x^{(i)T}x^{j}" title="max: L(\alpha )=\sum_{i=1}^{n}\alpha_{i} - \frac{1}{2}\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}label^{(i)}label^{(j)}x^{(i)T}x^{j}" /></a>  (13)

现在优化目标就转化为求解alpha。约束条件为：

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;\geqslant&space;0,\sum_{i=1}^{n}\alpha_{i}label^{(i)}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;\geqslant&space;0,\sum_{i=1}^{n}\alpha_{i}label^{(i)}=0" title="\alpha \geqslant 0,\sum_{i=1}^{n}\alpha_{i}label^{(i)}=0" /></a>

对于数据不是100%可分的情形，引入松弛变量C，来允许某些数据位于分割面错误的一侧。此时的约束条件就变化为：
 
<a href="https://www.codecogs.com/eqnedit.php?latex=C\geqslant&space;\alpha&space;\geqslant&space;0,\sum_{i=1}^{n}\alpha_{i}label^{(i)}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C\geqslant&space;\alpha&space;\geqslant&space;0,\sum_{i=1}^{n}\alpha_{i}label^{(i)}=0" title="C\geqslant \alpha \geqslant 0,\sum_{i=1}^{n}\alpha_{i}label^{(i)}=0" /></a>

接下来的问题就是通过样本来求解alpha了。SMO（sequence minimal optimization）就是训练求解alpha的算法。

SMO算法：
具体的公式推导参考
https://www.cnblogs.com/pinard/p/6111471.html。

这里直接抄结论放在这里。
SMO算法总结：
输入是m个样本(x1,y1),(x2,y2),...,(xm,ym),,其中x为n维特征向量。y为二元输出，值为1，或者-1.精度e。

输出是近似解α
1)取初值<a><img src="https://latex.codecogs.com/gif.latex?\alpha^{0}&space;=&space;0,&space;k&space;=0"/></a>

2)首选选择<a><img src="https://latex.codecogs.com/gif.latex?\alpha_1^{k}"/></a>,选择<a><img src="https://latex.codecogs.com/gif.latex?\alpha_2^{k}"/></a>，求出新的<a><img src="https://latex.codecogs.com/gif.latex?\alpha_2^{new,unc}"/></a>。

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_2^{new,unc}&space;=&space;\alpha_2^{k}&space;&plus;&space;\frac{y_2(E_1-E_2)}{K_{11}&space;&plus;K_{22}-2K_{12})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_2^{new,unc}&space;=&space;\alpha_2^{k}&space;&plus;&space;\frac{y_2(E_1-E_2)}{K_{11}&space;&plus;K_{22}-2K_{12})}" title="\alpha_2^{new,unc} = \alpha_2^{k} + \frac{y_2(E_1-E_2)}{K_{11} +K_{22}-2K_{12})}" /></a>

3)按照下式求出<a><img src="https://latex.codecogs.com/gif.latex?\alpha_2^{k+1}"/></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_2^{k&plus;1}=&space;\begin{cases}&space;H&&space;{L&space;\leq&space;\alpha_2^{new,unc}&space;>&space;H}\\&space;\alpha_2^{new,unc}&&space;{L&space;\leq&space;\alpha_2^{new,unc}&space;\leq&space;H}\\&space;L&&space;{\alpha_2^{new,unc}&space;<&space;L}&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_2^{k&plus;1}=&space;\begin{cases}&space;H&&space;{L&space;\leq&space;\alpha_2^{new,unc}&space;>&space;H}\\&space;\alpha_2^{new,unc}&&space;{L&space;\leq&space;\alpha_2^{new,unc}&space;\leq&space;H}\\&space;L&&space;{\alpha_2^{new,unc}&space;<&space;L}&space;\end{cases}" title="\alpha_2^{k+1}= \begin{cases} H& {L \leq \alpha_2^{new,unc} > H}\\ \alpha_2^{new,unc}& {L \leq \alpha_2^{new,unc} \leq H}\\ L& {\alpha_2^{new,unc} < L} \end{cases}" /></a>

4)利用<a><img src="https://latex.codecogs.com/gif.latex?\alpha_2^{k+1}"/></a>和<a><img src="https://latex.codecogs.com/gif.latex?\alpha_1^{k+1}"/></a>的关系求出<a><img src="https://latex.codecogs.com/gif.latex?\alpha_1^{k+1}"/></a>

5)按照公式计算<a><img src="https://latex.codecogs.com/gif.latex?\b^{k+1}"/></a>和Ei


6）在精度e范围内检查是否满足如下的终止条件：

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum\limits_{i=1}^{m}\alpha_iy_i&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum\limits_{i=1}^{m}\alpha_iy_i&space;=&space;0" title="\sum\limits_{i=1}^{m}\alpha_iy_i = 0" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=0&space;\leq&space;\alpha_i&space;\leq&space;C,&space;i&space;=1,2...m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?0&space;\leq&space;\alpha_i&space;\leq&space;C,&space;i&space;=1,2...m" title="0 \leq \alpha_i \leq C, i =1,2...m" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{i}^{k&plus;1}&space;=&space;0&space;\Rightarrow&space;y_ig(x_i)&space;\geq&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{i}^{k&plus;1}&space;=&space;0&space;\Rightarrow&space;y_ig(x_i)&space;\geq&space;1" title="\alpha_{i}^{k+1} = 0 \Rightarrow y_ig(x_i) \geq 1" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=0&space;<\alpha_{i}^{k&plus;1}&space;<&space;C&space;\Rightarrow&space;y_ig(x_i)&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?0&space;<\alpha_{i}^{k&plus;1}&space;<&space;C&space;\Rightarrow&space;y_ig(x_i)&space;=&space;1" title="0 <\alpha_{i}^{k+1} < C \Rightarrow y_ig(x_i) = 1" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{i}^{k&plus;1}=&space;C&space;\Rightarrow&space;y_ig(x_i)&space;\leq&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{i}^{k&plus;1}=&space;C&space;\Rightarrow&space;y_ig(x_i)&space;\leq&space;1" title="\alpha_{i}^{k+1}= C \Rightarrow y_ig(x_i) \leq 1" /></a>

7)如果满足则结束，返回<a><img src="https://latex.codecogs.com/gif.latex?\alpha^{k+1}"/></a>,否则转到步骤2）。

## 核函数
上面讨论的时线性可分时的情况，那么线性不可分的情况是不是意味着svm失灵了呢？我们知道低维不可分，在高维度下是线性可分的，如果把低维度数据映射到高维度，那么svm还是一样可以继续发挥作用了。
具体的参考http://www.cnblogs.com/pinard/p/6103615.html，讲的比较好。