(= (is-active $customer)
   (AccountStatus $customer Active))

(= (has-login-problem $customer)
   (HasIssue $customer LoginFailure))