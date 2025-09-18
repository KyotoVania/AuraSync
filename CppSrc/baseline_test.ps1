# Baseline Performance Test Script for BPM Analysis
# Tests all audio files in tests/assets and collects BPM results

$testFiles = @(
    @{file="Atlas160BpmF#G.wav"; expected=160},
    @{file="BenJMuwashshaMaster155BpmFm.wav"; expected=155},
    @{file="BélavieLost Woods152BpmAm.wav"; expected=152},
    @{file="Just Want To Ug You!150BpmBm.wav"; expected=150},
    @{file="LukeNoizePumpIt160BpmCm.wav"; expected=160},
    @{file="SimonTheDivergentStar116BpmFm.wav"; expected=116},
    @{file="TsukisayuYoru163BpmF#G.wav"; expected=163},
    @{file="wanzboreddd160BpmE.wav"; expected=160},
    @{file="song.wav"; expected=-1}  # Unknown BPM
)

$results = @()
$exe = "cmake-build-debug\bin\ave_analysis.exe"

Write-Host "=== BPM Baseline Performance Test ===" -ForegroundColor Green
Write-Host "Testing hybrid implementation across corpus..." -ForegroundColor Cyan

foreach ($test in $testFiles) {
    $filePath = "tests\assets\$($test.file)"
    $expected = $test.expected
    
    Write-Host "`nTesting: $($test.file)" -ForegroundColor Yellow
    if ($expected -gt 0) {
        Write-Host "Expected BPM: $expected" -ForegroundColor Gray
    }
    
    # Run analysis and capture output
    $output = & $exe $filePath 2>&1
    
    # Extract BPM from output
    $bpmLine = $output | Select-String "BPM: ([\d.]+)"
    if ($bpmLine) {
        $detectedBPM = [math]::Round([double]$bpmLine.Matches[0].Groups[1].Value, 2)
        
        # Extract confidence
        $confLine = $output | Select-String "confidence: ([\d.]+)"
        $confidence = if ($confLine) { [math]::Round([double]$confLine.Matches[0].Groups[1].Value, 4) } else { 0 }
        
        # Calculate error metrics
        if ($expected -gt 0) {
            $errorAbs = [math]::Abs($detectedBPM - $expected)
            $errorPercent = [math]::Round(($errorAbs / $expected) * 100, 2)
            
            # Classify result
            $classification = "SUCCESS"
            if ($errorPercent -gt 10) {
                # Check for octave errors
                $halfExpected = $expected / 2
                $doubleExpected = $expected * 2
                
                if ([math]::Abs($detectedBPM - $halfExpected) / $halfExpected * 100 -lt 5) {
                    $classification = "OCTAVE_ERROR_HALF"
                } elseif ([math]::Abs($detectedBPM - $doubleExpected) / $doubleExpected * 100 -lt 5) {
                    $classification = "OCTAVE_ERROR_DOUBLE"
                } else {
                    $classification = "FAILURE"
                }
            }
            
            Write-Host "Detected: $detectedBPM BPM (error: $errorPercent%)" -ForegroundColor $(if($errorPercent -lt 5){"Green"}elseif($errorPercent -lt 10){"Yellow"}else{"Red"})
            Write-Host "Confidence: $confidence" -ForegroundColor Gray
            Write-Host "Classification: $classification" -ForegroundColor $(if($classification -eq "SUCCESS"){"Green"}elseif($classification.StartsWith("OCTAVE")){"Yellow"}else{"Red"})
            
        } else {
            Write-Host "Detected: $detectedBPM BPM" -ForegroundColor Cyan
            Write-Host "Confidence: $confidence" -ForegroundColor Gray
            $classification = "UNKNOWN_REFERENCE"
            $errorPercent = -1
        }
        
        $results += @{
            File = $test.file
            Expected = $expected
            Detected = $detectedBPM
            Confidence = $confidence
            Error = if($expected -gt 0){$errorPercent}else{-1}
            Classification = $classification
        }
    } else {
        Write-Host "ERROR: Could not extract BPM from output" -ForegroundColor Red
        $results += @{
            File = $test.file
            Expected = $expected
            Detected = -1
            Confidence = 0
            Error = -1
            Classification = "PARSE_ERROR"
        }
    }
}

# Generate summary report
Write-Host "`n=== BASELINE PERFORMANCE SUMMARY ===" -ForegroundColor Green

$knownBPMTests = $results | Where-Object { $_.Expected -gt 0 }
$successes = $knownBPMTests | Where-Object { $_.Classification -eq "SUCCESS" }
$octaveErrors = $knownBPMTests | Where-Object { $_.Classification.StartsWith("OCTAVE") }
$failures = $knownBPMTests | Where-Object { $_.Classification -eq "FAILURE" }

Write-Host "Total files with known BPM: $($knownBPMTests.Count)" -ForegroundColor Cyan
Write-Host "Successes: $($successes.Count) ($([math]::Round($successes.Count/$knownBPMTests.Count*100,1))%)" -ForegroundColor Green
Write-Host "Octave errors: $($octaveErrors.Count) ($([math]::Round($octaveErrors.Count/$knownBPMTests.Count*100,1))%)" -ForegroundColor Yellow
Write-Host "Failures: $($failures.Count) ($([math]::Round($failures.Count/$knownBPMTests.Count*100,1))%)" -ForegroundColor Red

$avgConf = ($knownBPMTests | ForEach-Object { $_.Confidence } | Measure-Object -Average).Average
Write-Host "`nAverage confidence: $([math]::Round($avgConf, 3))" -ForegroundColor Cyan

Write-Host "`n=== DETAILED RESULTS ===" -ForegroundColor Green
foreach ($result in $results) {
    $color = switch ($result.Classification) {
        "SUCCESS" { "Green" }
        {$_.StartsWith("OCTAVE")} { "Yellow" }
        "FAILURE" { "Red" }
        default { "Gray" }
    }
    Write-Host "$($result.File): $($result.Detected) BPM (expected: $($result.Expected), confidence: $($result.Confidence), $($result.Classification))" -ForegroundColor $color
}

Write-Host "`n=== TEST COMPLETE ===" -ForegroundColor Green