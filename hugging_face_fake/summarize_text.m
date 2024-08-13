function hn_summaries = summarize_text(data)
    try


         package_path = 'C:\Users\연준모\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers'; % 여기에 패키지가 설치된 경로를 입력하세요
        if count(py.sys.path, package_path) == 0
            insert(py.sys.path, int32(0), package_path);
        end

        % Python 모듈 import
        py.importlib.import_module('transformers');
        py.importlib.import_module('torch');

        


        % 허깅페이스 모델 로드
        model_name = 'sshleifer/distilbart-cnn-12-6'; % 예시 모델 (DistilBART)
        tokenizer = py.transformers.AutoTokenizer.from_pretrained(model_name);
        model = py.transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name);

        % 데이터 처리 및 요약
        hn_summaries = cell(length(data), 1);
        for i = 1:length(data)
            title = data(i).title;
            try
                summary = generate_summary_py(tokenizer, model, title);
            catch ME
                summary = "Summary generation failed";
                disp(['Error occurred for title: ' title]);
                disp(ME.message);
            end
            hn_summaries{i} = char(summary);
        end

        % 결과 출력
        fprintf('\n=== Hacker News Top Stories ===\n');
        for i = 1:length(hn_summaries)
            fprintf('Title: %s\n', data(i).title);
            fprintf('Link: %s\n', data(i).link);
            fprintf('Summary: %s\n', hn_summaries{i});
            fprintf('----------------------------------\n');
        end

    catch ME
        disp('Error importing Python modules:');
        disp(ME.message);
        hn_summaries = [];
    end
end