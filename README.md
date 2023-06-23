textogram
====

Text-based histograms.


Example usage
----

Here's me finding out how long my commit titles are.

    $ git log --format=%s | awk '{print length}' | textogram

      4 - 15  : ###################                                 (97)
     15 - 26  : ##################################################  (248)
     26 - 37  : ####################################                (182)
     37 - 48  : ########################                            (123)
     48 - 58  : ######                                              (31)
     58 - 69  :                                                     (1)
     69 - 80  :                                                     (0)
     80 - 91  :                                                     (0)
     91 - 102 :                                                     (0)
    102 - 113 :                                                     (3)

Now let me focus in on the meat of the distribution and make the bin edges tidy.

    $ git log --format=%s | awk '{print length}' | textogram -a 0 -z 60 -n 12

     0 - 5  :                                                     (1)
     5 - 10 : ############                                        (30)
    10 - 15 : ###########################                         (66)
    15 - 20 : ##################################################  (118)
    20 - 25 : #################################################   (116)
    25 - 30 : #####################################               (89)
    30 - 35 : ###############################                     (74)
    35 - 40 : ################################                    (76)
    40 - 45 : ########################                            (58)
    45 - 50 : ##################                                  (44)
    50 - 55 : ###                                                 (9)
    55 - 60 :                                                     (0)
        >60 : (4)

We can also plot it with logarithmic y-axis.

    $ git log --format=%s | awk '{print length}' | textogram -a 0 -z 60 -n 12 -y log

     0 - 5  : ######                                              (1)
     5 - 10 : #####################################               (30)
    10 - 15 : ############################################        (66)
    15 - 20 : ##################################################  (118)
    20 - 25 : #################################################   (116)
    25 - 30 : ###############################################     (89)
    30 - 35 : #############################################       (74)
    35 - 40 : #############################################       (76)
    40 - 45 : ###########################################         (58)
    45 - 50 : ########################################            (44)
    50 - 55 : ##########################                          (9)
    55 - 60 :                                                     (0)
        >60 : (4)


Installation
----

    $ python setup.py install
