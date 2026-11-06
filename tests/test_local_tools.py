import unittest
from datetime import datetime, timezone
from kelus.local_tools import (CHECK_IN_REPLY, CRISIS_REPLY, DANGER_REPLY, LOSS_REPLY, OTHER_PERSON_REPLY, STILL_HERE_REPLY, calculate,
                               local_answer, support_follow_up)

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

    def test_arithmetic_wording_and_casual_suffixes_use_the_calculator(self):
        for prompt, expected in [
            ('whats 1+1 heh', '1+1 = 2'),
            ('what is fourty five plus five', '45 + 5 = 50'),
            ('what is forty-five plus 5?', '45 + 5 = 50'),
            ('how much is one hundred and five minus six', '105 - 6 = 99'),
            ('could you please calculate negative five times three', '-5 * 3 = -15'),
            ('calculate nine divided by zero', 'Division by zero is undefined.'),
            ('6 × 7 lol', '6 * 7 = 42')]:
            self.assertEqual(local_answer(prompt), (expected, 'local_calculator'), prompt)
        for prompt in ['write a story about forty five plus five', 'explain why 1+1 matters',
                       'what is twenty eleven plus one', 'what is five plus a puppy',
                       'write JavaScript for two times three', 'I scored 45+5 today heh']:
            self.assertIsNone(local_answer(prompt), prompt)
    def test_crisis_wording_always_gets_the_fixed_support_reply(self):
        from data.make_feelings import MORE_CRISIS, SITUATIONS
        for prompt in SITUATIONS['crisis'][1] + MORE_CRISIS + ["hi... I'm thinking about ending my life tonight", 'I feel like I want to die lol', 'Self-harm again :(']:
            self.assertEqual(local_answer(prompt), (CRISIS_REPLY, 'crisis_support'), prompt)
        for prompt in ['how do I kill a process in linux', 'this homework is killing me', "i'm dying to see that movie", 'my phone died',
                       'how do I end a for loop', 'cut the string in half', 'self-hosting a website', 'i broke up with my girlfriend']:
            self.assertIsNone(local_answer(prompt), prompt)
    def test_support_replies_fit_who_is_in_danger(self):
        expected = {
            CRISIS_REPLY: ['i wanna die', 'i wish i could sleep and never wake up', 'i wish i was never born', 'nothing to live for',
                           'im tired of being alive', 'i want to end it', 'im gonna hang myself', 'gonna kms',
                           'i got dumped and i want to die', 'my girlfriend left me and i feel suicidal',
                           'i cut myself shaving and i want to die', 'nobody would notice if i disappeared'],
            DANGER_REPLY: ['i took too many pills', 'i swallowed a bunch of sleeping pills', 'i just overdosed', "i'm not safe"],
            OTHER_PERSON_REPLY: ['my friend told me she wants to kill herself', 'my sister is cutting herself', 'my friend is suicidal',
                                 'my boyfriend said he wants to die'],
            LOSS_REPLY: ['my brother killed himself last year', 'i lost my dad to suicide'],
            CHECK_IN_REPLY: ['whats the point of living', "i can't do this anymore", 'i hate my life', 'i hate myself', 'my life is over',
                             "i'm a burden to everyone", 'i should just disappear', "i just don't see the point anymore"],
        }
        for reply, prompts in expected.items():
            for prompt in prompts:
                self.assertEqual(local_answer(prompt), (reply, 'crisis_support'), prompt)
        for prompt in ['my feet are killing me', 'i could die for a pizza', "i'm gonna die of embarrassment", 'i want to end it with my boyfriend',
                       'what is the point of this function', "i can't do this homework", 'i hate my job', 'i hurt myself playing soccer',
                       'i cut myself shaving', 'my friend hurt herself playing soccer', 'how far is 200 kms', 'what is suicide squad about',
                       'he killed it on stage', 'my mom has cancer', 'kill myself laughing lol']:
            self.assertIsNone(local_answer(prompt), prompt)
    def test_one_fixed_reply_follows_a_crisis_reply_unless_the_subject_changes(self):
        def chat(*follow, first=CRISIS_REPLY):
            return [{'role': 'user', 'content': 'i want to die'}, {'role': 'assistant', 'content': first}, {'role': 'user', 'content': follow[0]}]
        for follow in ['nobody cares', 'thanks', 'what should i do', 'why', 'ok']:
            self.assertEqual(support_follow_up(chat(follow)), STILL_HERE_REPLY, follow)
        self.assertEqual(support_follow_up(chat('ok', first=DANGER_REPLY)), STILL_HERE_REPLY)
        for follow in ["what's the capital of france?", 'write a python function that adds numbers']:
            self.assertIsNone(support_follow_up(chat(follow)), follow)
        self.assertIsNone(support_follow_up(chat('thanks', first=STILL_HERE_REPLY)))
        self.assertIsNone(support_follow_up([{'role': 'user', 'content': 'hi'}]))
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
