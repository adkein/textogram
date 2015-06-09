textogram
====

Text-based histograms.


Example usage
----

Here's how we might generate some random numbers:

    $ python -c 'import numpy; print "\n".join(map(str, numpy.random.normal(0, 1, 10)))'

    -0.583153545587
    0.56449538622
    0.458242637764
    -1.19546135709
    -1.15370979339
    0.559653304184
    -1.17324913199
    0.332889355294
    -1.03958229017
    -0.117342752587

Here's how we can make a textogram of, say, 1,000 such values:

    $ python -c 'import numpy; print "\n".join(map(str, numpy.random.normal(0, 1, 1000)))' | python -m textogram

     -3.6 -  -2.9: 
     -2.9 -  -2.2: 
     -2.2 -  -1.5: ###
     -1.5 -  -0.9: ###########
     -0.9 -  -0.2: ####################
     -0.2 -   0.5: ###################
      0.5 -   1.1: #############
      1.1 -   1.8: #######
      1.8 -   2.5: #
      2.5 -   3.2: 

    item count = 1000
    max_height_value = 256

We can get crazy and scale the y-axis logarithmically:

    $ python -c 'import numpy; print "\n".join(map(str, numpy.random.normal(0, 1, 1000)))' | python -m textogram -y log

     -3.1 -  -2.4: ########
     -2.4 -  -1.8: ############
     -1.8 -  -1.1: ################
     -1.1 -  -0.5: ###################
     -0.5 -   0.1: ####################
      0.1 -   0.8: ###################
      0.8 -   1.4: #################
      1.4 -   2.1: ###############
      2.1 -   2.7: #########
      2.7 -   3.3: ######

    item count = 1000
    max_height_value = 9
