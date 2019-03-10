# Gotham-City-Cabs

It is around the year 2034 in the city of Gotham, and the last time Batman got into a fight
with the Joker, the Batmobile (Batman's high-tech car) was seriously damaged. Apparently,
it would take his butler, Alfred, a while to fix the car and during that time Batman needs
to use a cab to save the people of the city!

Alfred needs your help to come up with a good prediction of the taxi trip duration
between multiple points of the Gotham city. If he can make such predictions, then that
significantly helps with Batman's missions.

Lucius (Batman's tech support guy) has been able to pull out a rich dataset of the
recorded taxi durations between various parts of the city and is sharing that with you for
your modeling purposes.

The input features of the aforementioned data file are:

pickup datetime: a variable containing a date and a time specifying the date and the
time the taxi picked of a passenger. For instance, you may observe a pickup datetime
of \6/14/2034 3:00:00 AM", which indicates the time the taxi picked up the passenger.

pickup x: This is a variable that represents the x coordinate of the location the taxi
picked up the passenger.

pickup y: This is a variable that represents the y coordinate of the location the taxi
picked up the passenger.

dropoff x: This is a variable that represents the x coordinate of the location the tax
dropped off the passenger.

dropoff y: This is a variable that represents the y coordinate of the location the taxi
dropped off the passenger.

The response variable is:

duration: which is the duration of the trip in seconds.
