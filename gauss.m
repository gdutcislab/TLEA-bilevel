function p = gauss(mu, sigma)
    p= mu + sigma * sqrt(-2.0 * log(rand)) * sin(2.0 * pi * rand);
end