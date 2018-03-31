#!/bin/tcsh

cd /Volumes/My_HD/Dropbox/matlab/emergentproj/data

foreach group ( nullcontrol) #network control
    foreach id (11)#01 02 03 04 05 06 07 08 09 10
        set lines = `wc -l < ${group}${id}_trial.txt`
        
        @ lines = $lines - 1
        
        tail -n$lines < ${group}${id}_trial.txt > ${group}${id}_trial_noHeader.txt
        
        cut -d'	' -f11-70 ${group}${id}_trial_noHeader.txt > ${group}${id}_trial_layers.txt
        cut -d'	' -f5 ${group}${id}_trial_noHeader.txt > ${group}${id}_trial_names.txt
        cut -d'	' -f9 ${group}${id}_trial_noHeader.txt > ${group}${id}_trial_sse.txt
        
        head -1 ${group}${id}_trial_layers.txt | wc -w 
        
        tr -s 'BTSXVPE_->\"' '	' < ${group}${id}_trial_names.txt > ${group}${id}_trial_nodeId.txt
        
        sed 's/"B->T_/1\    /g' < ${group}${id}_trial_names.txt | \
        sed 's/"B->P_/2\    /g' | \
        sed 's/"S->S_/3\    /g' | \
        sed 's/"T->S_/4\    /g' | \
        sed 's/"T->X_/5\    /g' | \
        sed 's/"S->X_/6\    /g' | \
        sed 's/"T->T_/7\    /g' | \
        sed 's/"P->T_/8\    /g' | \
        sed 's/"X->T_/9\    /g' | \
        sed 's/"X->V_/10\    /g' | \
        sed 's/"T->V_/11\    /g' | \
        sed 's/"P->V_/12\    /g' | \
        sed 's/"X->X_/13\    /g' | \
        sed 's/"P->X_/14\    /g' | \
        sed 's/"X->S_/15\    /g' | \
        sed 's/"P->S_/16\    /g' | \
        sed 's/"V->P_/17\    /g' | \
        sed 's/"V->V_/18\    /g' | \
        sed 's/"V->E_/19\    /g' | \
        sed 's/"S->E_/20\    /g' | \
        sed 's/0->1"/1/g' | \
        sed 's/0->2"/2/g' | \
        sed 's/1->1"/3/g' | \
        sed 's/1->3"/4/g' | \
        sed 's/2->2"/5/g' | \
        sed 's/2->4"/6/g' | \
        sed 's/3->2"/7/g' | \
        sed 's/3->5"/8/g' | \
        sed 's/4->3"/9/g' | \
        sed 's/4->5"/10/g' | \
        sed 's/5->0"/11/g' \
        > ${group}${id}_trial_typeId.txt
        
        #trial_type, node_type, *node1*, node2, sse
        paste -d' ' ${group}${id}_trial_typeId.txt ${group}${id}_trial_nodeId.txt ${group}${id}_trial_sse.txt \
        | tr -s ' ' '\t' > ${group}${id}_trial_labels.txt
        end
end
