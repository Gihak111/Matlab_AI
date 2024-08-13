function reshaped_array = my_reshape(input_array, new_size_row, new_size_col)
    % Custom reshape function to reshape input_array
    % Check if new_size_col is empty and calculate automatically
    if isempty(new_size_col)
        new_size_col = numel(input_array) / new_size_row;
    end
    
    % Check if the total number of elements is consistent
    if prod(size(input_array)) ~= (new_size_row * new_size_col)
        error('Number of elements must remain constant in reshape operation');
    end
    
    % Reshape the array using MATLAB's reshape function
    reshaped_array = reshape(input_array, new_size_row, new_size_col);
end