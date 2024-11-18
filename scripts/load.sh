seq_len=168
model=GPT4TS

for percent in 100
do
for pred_len in 24 72 120 168
do
for lr in 0.0001
do

python main.py \
    --root_path ./datasets/ \
    --data_path RI_G4400010_load.csv \
    --model_id RI_G4400010_load_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 84 \
    --pred_len $pred_len \
    --batch_size 256 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --target electricity

done
done
done
