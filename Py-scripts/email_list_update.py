#!/usr/bin/env python3
#this phyton script is useful for updating company email domain
#code by LD-Shell
#first create a function that is reusable
def email_update(email, old_domain, new_domain):
    #print email to be modified
    print(email)
    #set a condition, if true, to modify the email
    if "@"+old_domain in email:
   #index the old domain to be replaced
      index=email.index("@")
      new_email=email[:index]+"@"+new_domain
      print(new_email)
    return email
