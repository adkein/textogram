textogram
====

Text-based histograms.


Example usage
----

Here's me finding out how long my commit titles are.

    $ git log --format=%s | awk '{print length}' | textogram

      4 -  15: ###################                                 (96.0)
     15 -  26: ##################################################  (247.0)
     26 -  37: ####################################                (180.0)
     37 -  48: ########################                            (123.0)
     48 -  58: ######                                              (31.0)
     58 -  69:                                                     (1.0)
     69 -  80:                                                     (0.0)
     80 -  91:                                                     (0.0)
     91 - 102:                                                     (0.0)
    102 - 113:                                                     (3.0)

Now let me focus in on the meat of the distribution and make the bin edges tidy.

    $ git log --format=%s | awk '{print length}' | textogram -a 0 -z 60 -n 12

    0 -  5:                                                     (1.0)
    5 - 10: ############                                        (30.0)
    10 - 15: ###########################                         (65.0)
    15 - 20: ##################################################  (118.0)
    20 - 25: ################################################    (115.0)
    25 - 30: #####################################               (89.0)
    30 - 35: ##############################                      (72.0)
    35 - 40: ################################                    (76.0)
    40 - 45: ########################                            (58.0)
    45 - 50: ##################                                  (44.0)
    50 - 55: ###                                                 (9.0)
    55 - 60:                                                     (0.0)
    >60.0: (4)

