# The FSL model

## Description

_In construction ..._

In the FSL model we consider subhalos as a population. To each subhalo we can associate intrisic quantities (mass $m$, concentration $c$, position inside the host $\vec{r}$, ...). We denote by the _vector_ $s = (m, c, \vec{r}, ...)$ the set of all these quantities and look at their evolution from an _Eulerian_ point of view. We assume that we know the probability density function for subhalos to have properties $s_0$ when accreted and to be accreted at redshift $z_0$. We denote that function by $p_{(s_0, z_0)}$ and we will specify it below. In other words, we know the probability for a subhalo that exists at redshift $z$ to have a mass $m_0$, a concentration $c_0$, ... and to be accreted at redshift $z_0$. The master equation at the centre of the FSL model relates this _initial_ probability distribution to the late time distribution at redshift $z$ with a _transfer_ and a _surviving_ function so that

$$ p_s(s \, | \, z) = \frac{1}{f_{\mathcal{S}}(z)} \iint \mathcal{I}(s_0, z_0) \mathcal{T}(s \, | \,s_0, z_0, z)  \mathcal{S}(s_0, z_0, z) {\rm d}s_0 {\rm d} z_0 \, .$$


- the _initial function_ $\mathcal{I}$ is the probability density function of having $s_0$ and $z_0$ 
- the _transfer function_ $\mathcal{T}$ is the probability density function of having $s$ at $z$ knowing $s_0$ and $z_0$ 
- the _survival function_ $\mathcal{S}$ is 1 if the subhalo with $s_0$ and $z_0$ exists/survives at $z$ and 0 otherwise. Be careful $\mathcal{S}$ is not a probability density function, which is the reason why we need to include a normalisation fraction $f_{\mathcal{S}}$. 

The _normalisation fraction_ or _fraction of surviving subhalos_ $f_{\mathcal{S}}$ is simply given by (asking that the integral over $s$ of $p_s$ is 1 and using the fact that $\mathcal{T}$ is a correctly normalised probability density function),

$$ f_{\mathcal{S}}(z) =  \iint \mathcal{I}(s_0, z_0)  \mathcal{S}(s_0, z_0, z) {\rm d}s_0 {\rm d} z_0 $$

Assuming that all subhalo survive the total number of subhalos today (at redshift $z=0$) is denoted by $N_0$. The true total number of subhalos $N(z)$ at redshift $z$ is thus

$$N(z) = f_{\mathcal{S}}(z)N_0$$

Said otherwise, the density of halos with properties $s$ at redshift $z$ are

$$N(s \, |  \, z) = N(z) p_s(s \, | \, z) =  N_0 \iint \mathcal{I}(s_0, z_0) \mathcal{T}(s \, | \,s_0, z_0, z)  \mathcal{S}(s_0, z_0, z) {\rm d}s_0 {\rm d} z_0 \, .$$

In the following we detail the different expressions of the functions $\mathcal{I}$, $\mathcal{T}$ and $\mathcal{S}$. 

### The initial function

The initial distribution of subalos _at accretion_ can be written as a product of probability density functions on the space spanned by the variable $s_0$. In practice we write
$$ \mathcal{I}(s_0, z_0)  = p_{(s_0, z_0)}(s_0, z_0) = p_{(m_0, z_0)}(m_0, z_0) p(c_0 \, | \, m_0, z_0) p_{\vec{r}_0}(\vec{r}_0 \, |\, z_0) \, .$$
The mass and accretion redshift density $p_{(m_0, z_0)}(m_0, z_0) = p_{m_0}(m_0 \, | \, z_0) p_{z_0}(z_0) $ is given by the subhalo mass function

$$ p_{m_0}(m_0, z_0) = \frac{1}{N_0} \frac{\partial^2 N}{\partial m_0 \partial z_0}(m_0, z_0) = \frac{p_{z_0}(z_0)}{N_0} \frac{\partial N}{\partial m_0} (m_0 \, | \, z_0) $$
where the normalisation factor is
$$ N_0 \equiv \iint \frac{\partial^2 N}{\partial m_0 \partial z_0} {\rm d} m_0 {\rm d}z_0 \, . $$



### The transfer function

We assume that we can follow the evolution of individual halo and trace the value of its properties $s$ at $z$ directly from $s_0$ and $z_0$. Said differently, there is a unique function $f$ that gives the value of $s$ from $s_0$, $z_0$ and $z$. Therefore the transfer function $\mathcal{T}$ takes the form of a Dirac distribution
$$ \mathcal{T}(s \, | \, s_0, z_0,  z) = p(s \, | \, s_0, z_0,  z) =  \delta[s - f(s_0, z_0, z)] \, .$$ 
Let use denote $\hat s_0(s, z_0, z)$ the value of $s_0$ such that $f(\hat s_0(s, z_0, z), z_0, z) = s$. Then, it is numerically convinient to first integrate over the variable $s_0$ to obtain
$$ p_s(s \, | \, z) = \frac{1}{f_{\mathcal{S}}(z)} \int \mathcal{I}(\hat s_0(s, z_0, z), z_0) \left|\frac{\partial f}{\partial s_0}\right|^{-1}_{f(s_0, z_0, z) = s}  \mathcal{S}(\hat s_0(s, z_0, z), z_0, z)  {\rm d} z_0 \, .$$

Note however that $s$ is a multidimensional variable and, consequently, $f$ is a multidimensional function. Therefore what appears as _the absolute value of a derivative_ in the latter expression is the determinant of a Jacobian matrix. 



### The surviving function

The surviving function takes a simple form

$$  \mathcal{S}(s_0, z_0, z) = \Theta[g(s_0, z_0, z)]\Theta[z_0 - z]$$


## Functions

```@docs
subhalo_mass_function_template(x::Real, γ1::Real,  α1::Real, γ2::Real, α2::Real, β::Real, ζ::Real)
```

```@docs
mass_function_merger_tree(mΔ_sub::Real, mΔ_host::Real) 
```
