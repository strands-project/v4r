function mapped = maplabels(origlbl, mapping)

mapped = origlbl;

for i=1:length(mapping)
    idx = (origlbl == i);
    mapped(idx) = mapping(i)-1;
end
