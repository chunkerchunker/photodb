# depends on jq module https://github.com/robzr/jq-printf
# download printf.jq to ~/.jq

find tmp/aws/batch-output -name "*manifest.json.out" -exec cat {} + \
  | jq -s 'include "printf";
           map({photos: .totalRecordCount, input: .inputTokenCount, output: .outputTokenCount})
           | reduce .[] as $i ({photos: 0, input:0, output:0};
               {photos: .photos + $i.photos, input: .input + $i.input, output: .output + $i.output})
           | . + {total_cost: (.input * 0.000125/1000 + .output * 0.000625/1000)}
           | { 
              input_tokens: .input,
              output_tokens: .output,
              total_photos: .photos,
              total_cost: .total_cost | printf("$%.2f"),
              "1k_photo_cost": (.total_cost / .photos * 1000) | printf("$%.2f"),
              photo_cost: (.total_cost / .photos * 100000 | round) / 10000
              }'
           