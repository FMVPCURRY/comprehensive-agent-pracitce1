$ErrorActionPreference = "Stop"

$python = "D:\anaconda\envs\nlp\python.exe"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Push-Location $root
try {
    & $python "prepare_dialogue_dataset.py"

    $bertLog = Join-Path $logDir "bert_train.log"
    $bertErr = Join-Path $logDir "bert_train.err.log"
    & $python "run.py" `
        --model "Bert" `
        --dataset-name "ChiFraudDialog" `
        --num-epochs 3 `
        --batch-size 16 `
        --pad-size 256 `
        --learning-rate 5e-5 `
        1>> $bertLog 2>> $bertErr

    $chineseLog = Join-Path $logDir "chinesebert_train.log"
    $chineseErr = Join-Path $logDir "chinesebert_train.err.log"
    & $python "run.py" `
        --model "Chinese_Bert" `
        --dataset-name "ChiFraudDialog" `
        --bert-path "./pretrained/ChineseBERT-base" `
        --num-epochs 3 `
        --batch-size 8 `
        --pad-size 256 `
        --learning-rate 5e-5 `
        1>> $chineseLog 2>> $chineseErr
}
finally {
    Pop-Location
}
