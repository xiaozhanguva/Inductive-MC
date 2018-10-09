function retVal=AveragePrecision(output,label)

[~,index]=sort(output,2,'descend');
[~,label_index]=sort(index,2,'ascend');

output_positive=output.*label-10000*(1-label);
[~,index_positive]=sort(output_positive,2,'descend');
[~,label_index_positive]=sort(index_positive,2,'ascend');
label_index_positive=label_index_positive.*label;

retVal=sum(label_index_positive./label_index,2)./sum(label,2);

retVal=mean(retVal);


end