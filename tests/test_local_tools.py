import unittest
from datetime import datetime, timezone
from kelus.local_tools import calculate, local_answer

class LocalToolTests(unittest.TestCase):
    def test_calculator_precedence_negative_decimals_and_limits(self):
        for expression, expected in [('1+1','2'), ('2+3*4','14'), ('(2+3)*4','20'), ('-4+2','-2'), ('0.5/2','0.25'), ('17%5','2'), ('7//2','3')]:
            self.assertEqual(calculate(expression),expected)
        for expression in ['__import__("os")', '2**999999', '1e999', '[1]*2', '9999999999999+1', '1+'*60+'1']:
            with self.assertRaises((ValueError, SyntaxError)):
                calculate(expression)
        self.assertEqual(local_answer('1/0?'),('Division by zero is undefined.','local_calculator'))
    def test_routing_is_anchored_and_source_is_honest(self):
        self.assertEqual(local_answer('what is 1+1?'), ('1+1 = 2','local_calculator'))
        self.assertEqual(local_answer('Calculate 6 × 7 please!'), ('6 * 7 = 42','local_calculator'))
        for prompt in ['write a function for 1+1', 'what is a date object', 'tell me about today in history', 'can you code?', 'nooo', 'print(1+1)', 'what is tomorrow\'s date', 'what time is it in Tokyo']:
            self.assertIsNone(local_answer(prompt))
    def test_clock_reads_injected_live_time_not_weights(self):
        now=datetime(2031,2,3,15,4,5,tzinfo=timezone.utc)
        text,source=local_answer('what date is today',now)
        self.assertEqual(source,'local_clock'); self.assertIn('February 3, 2031',text); self.assertIn('UTC',text)
        self.assertIn('15:04:05',local_answer('what time is it?',now)[0])


class RuntimeTests(unittest.TestCase):
    def test_history_policy_keeps_whole_recent_exchange_and_reports_omissions(self):
        from kelus.model import Model, Config
        from kelus.runtime import prepare_reply
        model=Model(Config(context=128,width=8,heads=2,layers=1))
        messages=[{'role':'user','content':'hi'},{'role':'assistant','content':'hello'},
                  {'role':'user','content':'variable'},{'role':'assistant','content':'a name'},
                  {'role':'user','content':'example'}]
        kept,omitted,tool=prepare_reply(model,messages)
        self.assertEqual(kept,messages[-3:]); self.assertEqual(omitted,2); self.assertIsNone(tool)
        messages[-3]['content']='1+1?'
        kept,omitted,_=prepare_reply(model,messages)
        self.assertEqual(kept,messages[-1:]); self.assertEqual(omitted,4)
        messages[-1]['content']='1+1?'
        self.assertEqual(prepare_reply(model,messages)[2],('1+1 = 2','local_calculator'))
        with self.assertRaises(ValueError): prepare_reply(model,[{'role':'assistant','content':'wrong'}])

class ToolHttpTests(unittest.TestCase):
    def test_http_labels_tool_answers_and_does_not_call_the_model(self):
        import http.client, json, threading
        from http.server import ThreadingHTTPServer
        from unittest.mock import patch
        from kelus.model import Model, Config
        from kelus.server import make_handler
        model=Model(Config(context=64,width=8,heads=2,layers=1))
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(model,{'config':{},'metadata':{}},'test.npz'))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        client=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
        try:
            with patch.object(Model,'generate',side_effect=AssertionError('Calculator must not sample model text')):
                client.request('POST','/api/chat',json.dumps({'messages':[{'role':'user','content':'1+1?'}]}),{'Content-Type':'application/json'})
                response=client.getresponse();self.assertEqual(response.status,200)
                events=[json.loads(line) for line in response.read().splitlines()]
                self.assertEqual(events[0],{'source':'local_calculator'})
                self.assertEqual(''.join(e.get('text','') for e in events),'1+1 = 2')
                self.assertTrue(events[-1]['done'])
        finally:
            client.close();server.shutdown();server.server_close();thread.join(2)

if __name__=='__main__': unittest.main()
