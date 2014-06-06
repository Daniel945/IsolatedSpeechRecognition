% main_dr_wav2mfcc_e_d_a;
% generate_selected_TI_isolated_digits_testing_list_mat;
% generate_selected_TI_isolated_digits_training_list_mat;

MODEL_NO=11;
dim=39;
feature_file_format='htk';
ITERATION_BEGIN=0; % Set ITERATION_BEGIN=0 to generate model structure
ITERATION_END=10;
A0=[0.8 0.2]; % the transition prob. from the dummy start state to the emitting states, i.e., A0(1) is used to initialize A(1,2) and A0(k) is used to initialize A(1,k+1) 
Aij=[0.6 0.3 0.1]; % the transition prob. from an emit-state to itself and the following states, i.e.,Aij(1) is used initialize A(i,i) and Aij(k) is used initialize A(i,i+k-1) for all i.
Af=[0.8 0.2];  % Af(k) is used to set the transition prob. from the last k-th emit-state to the null end state if Af(k) > A(N-k,N). For each k, if Af(k) is larger than A(N-k,N), then Af(k) is used to replace A(N-k,N) and the probability associated with the transition arcs leaving State k are renormalized. If Af(k) does not exist or Af(k) is not larger than A(N-k,N), then A(N-k,N) will not be affected.	
format compact;
for EMIT_STATE_NO=6:14
    N=EMIT_STATE_NO+2;
    model_filename_prefix='models/EM_models';
    traininglist_filename='training_list.mat';
    testinglist_filename='testing_list.mat'; % outside test
    accuracy_filename=sprintf('recog_rate_S%d_EM_models.mat',EMIT_STATE_NO);
    
    total_log_prob=zeros(1,ITERATION_END);
    recog_rate_EM_Viterbi=zeros(1,ITERATION_END);
    
    for iter=ITERATION_BEGIN:ITERATION_END
        model_filename_new=[model_filename_prefix, '_S', int2str(EMIT_STATE_NO), '_iter', int2str(iter),  '.mat'];
        if iter==0
            model_struct_filename=[model_filename_prefix, '_S', int2str(EMIT_STATE_NO), '_struct', int2str(iter),  '.mat'];
            %generate_LR_HMM_skips_structure(MODEL_NO,model_struct_filename,dim,N,A0,Aij,Af);
            %[global_mean_vec, global_var_vec, total_fr_no]=global_mean_var_for_hmm_skips_1gau(traininglist_filename,model_struct_filename, model_filename_new);
            generate_LR_HMM_skips_structure(MODEL_NO,model_filename_new,dim,N,A0,Aij,Af);
        else
            if ( exist(accuracy_filename,'file')  )
                load(accuracy_filename,'recog_rate_EM_Viterbi');
            end
            model_filename_old=[model_filename_prefix, '_S', int2str(EMIT_STATE_NO), '_iter', int2str(iter-1),  '.mat'];
            [total_log_prob(iter),total_fr_no,dim]=EM_hmm_skips_1gau(traininglist_filename,model_filename_old,model_filename_new);
            ave_log_prob_per_frame_per_dim=total_log_prob/total_fr_no/dim
            recog_rate_EM_Viterbi(iter)=recognition_Viterbi_hmm_skips_1gau(testinglist_filename,model_filename_new,feature_file_format)
            save(accuracy_filename,'recog_rate_EM_Viterbi');
        end
    end
end
