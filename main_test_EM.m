format compact;
feature_file_format='htk';
ITERATION_BEGIN=1; 
ITERATION_END=10;
for EMIT_STATE_NO=6:14
    N=EMIT_STATE_NO+2;
    model_filename_prefix='models/EM_models';
    testinglist_filename='testing_list.mat'; % outside test
    accuracy_filename=sprintf('recog_rate_S%d_EM_models.mat',EMIT_STATE_NO);
    
    recog_rate_EM_Viterbi=zeros(1,ITERATION_END);
    
    for iter=ITERATION_BEGIN:ITERATION_END
        model_filename_new=[model_filename_prefix, '_S', int2str(EMIT_STATE_NO), '_iter', int2str(iter),  '.mat'];
            if ( exist(accuracy_filename,'file')  )
                load(accuracy_filename,'recog_rate_EM_Viterbi');
            end
            recog_rate_EM_Viterbi(iter)=recognition_Viterbi_hmm_skips_1gau(testinglist_filename,model_filename_new,feature_file_format)
            save(accuracy_filename,'recog_rate_EM_Viterbi');
    end
end
