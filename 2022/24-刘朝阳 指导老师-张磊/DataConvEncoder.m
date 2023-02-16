

%in_bits = randn(1,2) > 0;
%in_bits = [0 1 0 1];
%obtained from output bits of DataViterbiDecoder.m
%in_bits = out_bits;
function convcoded_bits =  DataConvEncoder(in_bits)

ConvCodeGenPoly=[1 0 1 1 0 1 1;1 1 1 1 0 0 1];
number_rows = size(ConvCodeGenPoly, 1);
number_bits = size(ConvCodeGenPoly,2)+length(in_bits)-1;

uncoded_bits = zeros(number_rows, number_bits);

for row=1:number_rows
 uncoded_bits(row,1:number_bits)=rem(conv(double(in_bits), ConvCodeGenPoly(row,:)),2);
end

coded_bits=uncoded_bits(:);
convcoded_bits=coded_bits';

