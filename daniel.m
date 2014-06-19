% by Lee-Min Lee

infilename = 'Daniel/wav/test.wav';
outfilename = 'Daniel/1_test.mfc';

frame_size_sec = 0.025;
frame_shift_sec= 0.010;
use_hamming=1;
pre_emp=0;
bank_no=26;
cep_order=12;
lifter=22;

delta_win=2;
delta_win_weight = ones(1,2*delta_win+1);

[speech_raw, fs, bit_res]=wavread(infilename,'native');
speech_raw=double(speech_raw);
feature_seq=wav2mfcc_e_d_a(speech_raw,fs,frame_size_sec,frame_shift_sec,use_hamming,pre_emp,bank_no,cep_order,lifter,delta_win_weight);

[dim frame_no]=size(feature_seq);
  
fout=fopen(outfilename,'w','b'); % 'n'==local machine format 'b'==big endian 'l'==little endian
fwrite(fout,frame_no,'int32');
sampPeriod=round(frame_shift_sec*1E7);    
fwrite(fout,sampPeriod,'int32');
sampSize=dim*4;   
fwrite(fout,sampSize,'int16');
parmKind=838; % parameter kind code: MFCC=6, _E=64, _D=256, _A=512, MFCC_E_D_A=6+64+256+512=838
fwrite(fout,parmKind,'int16');  

% write data
fwrite(fout, feature_seq,'float32');


clear;
list_filename='daniel_testing_list.mat';
dir1='Daniel';
wordids={'1','2','3','4','5','6','7','8','9','O','Z'};

k=1;
list{k,1}=1;
list{k,2}=sprintf('%s/1_test.mfc', dir1);

save(list_filename,'list');


feature_file_format='htk';
format compact;

testinglist_filename='daniel_testing_list.mat'; % outside test

model_filename = 'models/EM_models_S8_iter6.mat';


load(testinglist_filename, 'list');
load(model_filename, 'mean_vec_i_m', 'var_vec_i_m', 'a_i_j_m');

[dim,N,MODEL_NO]=size(mean_vec_i_m);
correct_count=0;
error_count=0;
   
utterance_no=size(list,1);
for k=1:utterance_no
        filename=list{k,2};
        fid=fopen(filename,'r');
        if strcmpi(feature_file_format, 'HTK')
           fseek(fid, 12, 'bof'); % skip the 12-byte HTK header
        end
        c=fread(fid,'float','b');        
        fclose(fid);
        fr_no=length(c)/dim;
        c=reshape(c,dim,fr_no);
        scores=ones(MODEL_NO,1)*(-inf);
        for m=1:MODEL_NO
            scores(m)=viterbi_hmm_LR_skips_1gau( c, mean_vec_i_m(:,:,m), var_vec_i_m(:,:,m), a_i_j_m(:,:,m) );
        end
        [temp, m_max]=max(scores);
        fprintf('recog=%d \n',m_max);  
        if (m_max==list{k,1}) 
            correct_count=correct_count+1;
        else 
            error_count=error_count+1;
        end
end

total_count=(error_count+correct_count);
recog_rate=correct_count/total_count;
if nargout > 1
    varargout(1)= { total_count };
end