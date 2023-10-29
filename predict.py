from ensemble import *


if __name__ == '__main__':
    char_winning_candidates = [
        'model3_tone', 'model2_new', 'model35', 'model2', 'model5_synth', 'model15_synth_new', 'model30', 'model35_tone', 'model20', 'model13', 
        'model17', 'model31', 'model9', 'model29', 'model2_synth', 'model15_new', 'model26_synth', 'model18_new', 'model30_tone', 'model33_tone', 
        'model34', 'model19', 'model30_synth', 'model14', 'model5_synth_tone', 'model19_synth_new_tone', 'model8_synth', 'model35_synth', 'model27', 'model34_tone', 
        'model5_new', 'model15_synth_tone', 'model1', 'model5_tone', 'model20_tone', 'model3_synth', 'model15', 'model6_new', 'model1_synth', 'model19_synth_new'
    ]
    word_winning_candidates = [
        'model19_new', 'model3_tone', 
        'model35_tone', 
        'model30', 'model15_synth_new', 'model10_synth', 'model32', 'model8_synth', 
        'model5_synth_new', 'model5', 'model35_synth', 'model5_synth', 'model33_tone', 'model18_new', 'model31', 'model3_new', 
        'model10', 'model10_synth_new', 'model4_synth_new', 'model4_synth', 'model17_new', 'model7', 'model35', 'model2'
    ]
    char_winning_candidates = add_full_to_lst(char_winning_candidates)
    word_winning_candidates = add_full_to_lst(word_winning_candidates)

    char_based_pred_full = make_final_char_prediction(char_winning_candidates, set_name='private_test')
    pred = make_final_prediction(word_winning_candidates, char_based_pred_full, alpha=1.25, set_name='private_test')