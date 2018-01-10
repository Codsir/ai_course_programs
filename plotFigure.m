%% plot exp£¨cx£©-1

x =[ rand(1, 30)*0.5, rand(1, 30)*2];
y = [ rand(1, 30)*2, rand(1, 30)*0.5];
figure;
plot(x, y, '*');
xlabel('x');
ylabel('y');
title('before exp(cx)-1')

figure;
c= 1.2;
x= exp(c*x) - 1;
y = exp(c*y) - 1;
plot(x, y, '*r');
xlabel('x');
ylabel('y');
title('after exp(cx)-1')