function y = softmaxx(x)
    e_x = exp(x - max(x, [], 3));
    y = e_x ./ sum(e_x, 3);
    disp("소프트 매에에ㅔㅔ스")
end
