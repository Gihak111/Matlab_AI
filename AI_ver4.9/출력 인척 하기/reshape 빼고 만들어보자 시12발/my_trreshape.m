%{
function reshaped_array = my_trreshape(input_array, new_size_row, new_size_col)
    if isempty(new_size_col)
        new_size_col = numel(input_array) / new_size_row;
    end
    
    % Check if the total number of elements is consistent
    if prod(size(input_array)) ~= (new_size_row * new_size_col)
        error('Number of elements must remain constant in reshape operation');
    end
    
    % Reshape the array
    reshaped_array = zeros(new_size_row, new_size_col, 'like', input_array);
    
    % Linear indexing to reshape
    for i = 1:new_size_col
        for j = 1:new_size_row
            index = (i-1)*new_size_row + j;
            reshaped_array(j, i) = input_array(index);
        end
    end
end
%}
function Y = my_trreshape(X, dim1, dim2)
    totalElements = numel(X);
    assert(dim1 * dim2 == totalElements, 'Total number of elements must remain constant.');
    Y = reshape(X, [dim1, dim2]);
end
