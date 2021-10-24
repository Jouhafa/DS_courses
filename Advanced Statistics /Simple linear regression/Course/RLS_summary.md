---
title: 'Week 1 : Advanced Statistics & Regression'
created: '2021-10-21T20:35:43.872Z'
modified: '2021-10-23T09:44:17.732Z'
---

# Week 1 : Advanced Statistics & Regression

## Course 1 : Reminders & Simple Linear Regression 

### Reminders : 
*Data* : numeric, symbolic, mixtes, graphs, functions, texts ...

M : model, partially known $P_\theta$   
$\theta$ in $T$ :
  - So  if $T$ in $R^n$ : M is called *parametric*
  - otherwie, $T$ in an infinite-dem space, M is not parametric

**Goal** : from $x_1$, ..., $x_n$ and M we want to find $P_t$

- "Modèle identifiable" : if $P_\theta = P_\theta'$   $=>$ $\theta=\theta'$
- Statistic : every function of $X_i$
e.g : $S(Z)= (X_1 + ... + X_n)/ n$ 
- Probability Vs Statistics : 
  - In probability : we have $P_t$ and we calculate $(x_1, ..., x_n)$
  - In statistics, we try from $(x_1, ..., x_n)$ to find the $P_t$


### Simple Linear Regression 

In this part we are assuming that :   $Y = \beta_0 + \beta_1 X$
Terminology : 
- Y : Response
- X : Predictor
- $\beta_0$ : Interecept 
- $\beta_1$ : Slope 

The "Simplest" way to estimate $\beta_1$ and $\beta_0$ is using the LSE : Least Sequares Estimator 
$\hat{\beta_0}$, $\hat{\beta_1}$ $\in argmin \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2$

In the gaussian model : LSE $\sim$ MLE(Maximum Likelihood Estimator)

However, **LSE** is the best estimator that we can find within the not-biased estimators that lineary explain $Y_i$ in terms of the **minimum variance** 

$\hat{Y_i} = \hat{\beta_0} + \hat{\beta_1} X_i$

finally, we define residuals as :  $E_i = Y_i - \hat{Y_i}$

#### Hypothesis Tests

This part is dedicated to answer some questions that we may have concernign this model,such : is really the variable affect the response? How much it does affect it? 
So we transform these questions to hypothesis and try to evaluate them.
Let $H$ : variable affect the response 
So we annotate :
-  $H_0$ : $\beta_1 = 0$ (the variable has no effect on the response)
-  $H_1$ : $\beta_1 \neq 0$ (the variable has no effect on the response)

And we call $H_0$ the nul hypothesis.

So, to test the hypothesis we will need a statistical test and the distribution under this hypothesis.

For the case of gaussian residuals, and under the nul hypothesis, we have the statistical test $\frac{(n-2)R^2}{R^2}$ is following Fisher distribution 1,(n-2)

So, with a significativity level $\alpha$ we can **reject** $H_0$ if $F = \frac{R^2}{1-R^2} > f_{1; n-2; 1-\alpha/2}$

with $f_{1; n-2; 1-\alpha/2}$ is $1-\alpha/2$ quantile of Fisher distribution with (1, n-2) freedom degrees.


















