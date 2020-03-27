#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:30:42 2018

@author: akashteckchandani
"""

"""
DocString:

    
    A) Introduction:
    Welcome to this round of recruitment in The Coolest Company for the role of a DATA 
    ANALYST !! This round consists of three short questions based on which you  
    will be selected for the next round. All the best!
    note - all candidates who have applied for this job are from the following 
    backgrounds: Computer Science, Business, engineering, data science
    
    Question 1: Highest education qualifiacation
    Question 2: What was your master degree major in?
    Question 3: How many years of work experience do you have in this field?

    B) Known Bugs and/or Errors:
    None.

"""

from sys import exit

master_major = [ '1) Computer Science', '2) Business', '3) Engineering', \
'4) Data Science', '5) Other']

def game_start():
    global interviewer_name
    global candidate_name
    global country
    global city
    
    
    print("\nWelcome to this round of recruitment in The Coolest Company!!!\n")
    
    input('<Press enter to continue>\n')
    
    print("As you have applied for a job in my company, you better know my")
    
    print("name! It's... \n")
    interviewer_name = input('Input interviewer name: \n')
    
    print(f"\nYes, thank you, my name is {interviewer_name}!")
    
    print("""

Today we will interview our candidates who will compete for the DATA ANALYST
role!

What's your name candidate?

""")

    candidate_name = input('Input candidate name: \n')
    
    print(f"\nWelcome {candidate_name}!!!!\n")
    
    input('<Press enter to continue>\n')
    
    print(f"So to get to know you better {candidate_name}...")
    
    country = input("Which country are you from? \n>")
    
    city = input("Which city are you from? \n> ")
    
    print(f"""
          
Dear Ladies and gentlemen of the recruiting panel, this is {candidate_name},
Our candidate today is from {city},{country}.
May the most deserving get the job.

""")

    input('<Press enter to continue>\n')
    
    question_1()
    

def question_1():
    
    print(f"""

{interviewer_name}: Welcome to this round of recruitment in \
The Coolest Company!
So, our first round is about your Education Qualification!
   
""")
    input('<Press enter to continue>\n')

    
    print(f"""
          
{interviewer_name}: {candidate_name}, what is the highest level of education
you have completed?

    1) high school
    2) bachelor degree
    3) masters degree
    4) phd

""")
    edu_qua = input("> \n")
        
    if "3" in edu_qua or "masters degree" in edu_qua or "4" in edu_qua \
    or "phd" in edu_qua:
        
        print(f"""
              
{interviewer_name}: Congratulations!! you now move on to the next question!

""")
        input('<Press enter to continue>\n')
        question_2()
    
    else:
        print(f"""
              
{interviewer_name}: You are not yet qualified for this job. 
Go to college {candidate_name}!!!

""")
            
    input('<Press enter to continue>\n')
    
    fail()
    
def question_2():
    
    print(f"""
{interviewer_name}: So, {candidate_name}, what was your master degree major in?

""")

    for elements in master_major : 
        print (elements)
    
    mas_maj = input("> \n")
    if int(mas_maj) == 1 or int(mas_maj) == 2 or int(mas_maj) == 3 \
    or int(mas_maj) == 4 : 
    
        print(f"""
              
{interviewer_name}: Congratulations on making it through that!! It must \ 
have been one hell of a ride! You now move on to the next question!

""")
    
        input('<Press enter to continue>\n')
    
        question_3()
    else:
        print(f"""
              
{interviewer_name}: Your major does not match the specifications for this \
round of recruitment.
Sorry {candidate_name}. Please try again in another round.

""")

    input('<Press enter to continue>\n')
    fail()
    
def question_3():
    
    print(f"""
{interviewer_name}: How many years of work experience do you have in this field?

""")
    
    workex = input("> \n")
    if int(workex) >= 3 :
        
        print(f"""            
{interviewer_name}: 
  _  __      __  __  _ ____        _ ____  __      __ 
 / )/  )/| )/ _ /__)/_| /  /  //  /_| /  //  )/| )(   
(__(__// |/(__)/ ( (  |(  (__/(__(  |(  ((__// |/__)  
                                                      
  You have successfully cleared this \
round of recruitment! You will be contacted with further details! 
""")
        
    else:
        print(f"""
              
{interviewer_name}: Your experience does not match the specifications for this \
round of recruitment.
Sorry {candidate_name}. Please try again in another round.

""")
        
    
        fail()  
        
def fail():
    print(f"""
{interviewer_name}: Oh, I'm sorry {candidate_name}, looks like you do not 
qualify for the job :(
             
""")
    exit(0)


###############################################################################
# Game Start
###############################################################################
game_start()

