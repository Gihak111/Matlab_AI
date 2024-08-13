% 소프트맥스 함수 정의
function y = softmaxx(x)
    e_x = exp(x - max(x, [], 3));
    y = e_x ./ sum(e_x, 3);
end