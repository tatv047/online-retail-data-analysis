

There are questions that you can ask:

1. Who are your customers?
2. What do they buy together?
3. How are we doing inkeeping customers coming back?

Question (1) can be answered using: RFM based customer sgementation <br>
Question (2) can be answered using: Market Basket ANalysis <br>
Question (3) can be answered using: Cphort and Retention analysis. <br>

And that is exactly what we will do:
but befopre that we will need data exploration and data cleaning.



## Customer Lifetime Value Prediction

**Customer lifetime value** is all the gains, typically expressed in monetary terms, that a business gets from a particular customer throughout their lifetime. <br>
A customer’s lifetime is understood as the time during which they engage with the company. Put simply, it is the time between the customer’s first and last purchase ever. <br>
In order to calculate the CLV for a particular client, we need to predict the volume of products or services they will have bought before they churn and multiply it with some monetary figure, such as the price (this will result in CLV expressed as income per client) or profit margin (which will have our CLV estimate profit per client).

### CLV in non-contractual setting

- When the relationship between the company and its customer is formalized in the form of a written agreement, or contract, we are in the **contractual setting**. 
- Think about your Netflix subscription. Netflix knows the exact dates when you subscribed and canceled the subscription, that is your lifetime. They also have a lot of other data about you: what movies you watched and how you rated them, what times during the day were you watching the most, likely also your age and gender, etc. Based on all these data it is fairly straightforward to build a machine learning model predicting one’s churn probability.
- Now think about an online store you recently made a purchase at. You did not sign any contract with them, which makes it impossible for them to know for sure whether you have already churned or not. You might decide to buy with them again tomorrow or in ten years. Also, they don’t have much information about you other than your purchase history (especially if you are buying online as a guest). This is a **non-contractual setting**. In this case, your zillion-parameter mega-transformer will be of little use, since you cannot train a supervised machine learning model without the ground-truth labels. This is why we need a different approach.

### Buy Till You Die models

- The most popular and accurate way to estimate the CLV for non-contractual data that has been extensively used for years is the class of probabilistic models dubbed ‘**Buy Till You Die**’. One of them in particular, the **BG/NBD Gamma-Gamma model**, has seen widespread adoption.
- These are, in fact, two separate models:
    1. **BG/NBD** predicts the *future number of transactions per customer*, which is then fed into the Gamma-Gamma model.
    2. The **Gamma-Gamma** model to predict their *monetary value*.

### Beta Geometric / Negative Binomial Distribution Model

- The BG/NBD model predicts the *future number of purchases made by a customer* as well as their *probability of being alive*, i.e. not churned yet, in a non-contractual setting.
- This model is based on transaction data only. This means that for each customer, we need to know the time, volume, and value of each of their purchases.
- Using this we need to calcuate four distinct quantities for each customer:
    1. **Recency(R) :** the time between customer's last purchase date and end date of analysis.
    2. **Frequency(F) :** the count of time periods the customer made a purchase in (not the no. of repeat purchases the customer has made)
    3. **Monetary(M) :** monetary value or the average revenue or the income from the customer's repeat purchases.
    4. **T :** the customer’s age (this is the time difference between the customer’s first purchase and the end date of our analysis, often the last date available in the data)
- We will be using the **lifetimes** python package for this.

- The model assumes that as long as customer is alive,their number of transactions follows a **Poisson Process** with some rate **$\lambda$**. For instance, **$\lambda$** = 2 means that a cutsomer makes two purchases per time period on average. The rate **$\lambda$** is different for each customer,and its dustribution over all customers is assumed to be the **Gamma Distribution**. When the customer is dead(that is they churned), the transactions are obviouslly zero.
- After each purchase, a customer might churn with some **probability $p$**. This probability is different for each customer,and its distribution over all customers is assumed to be the **Beta Distribution.**
