from transformers import pipeline # pip install transformers

def abstractive_summarize(text, max_length=500):
    summarize_pipeline = pipeline('summarization')
    summaries = []

    # Create initial batches of text
    summary_stack = [text]

    # Loop until there is only 1 summary left
    epoch = 1
    while True:
        while summary_stack != []:
            # Get the first batch of text and try to summarize it
            # Note: If there are too many tokens the summarizer can't handle it, so we will break it into smaller batches if it fails
            text_batch = summary_stack.pop()
            try:
                # Append the summary of a batch
                max_length_final = min(max_length, len(text_batch.split()))
                min_length_final = max_length_final // 2
                summary = summarize_pipeline(text_batch, max_length=max_length_final, min_length=min_length_final, do_sample=False)
                summaries.append(summary[0]['summary_text']) # output format is a list of dictionaries [{'summary_text': ''}]
            except:
                # Put current batch back in the stack, split each batch and append to temporary stack, then set real stack back to temporary
                tmp_summary_stack = []
                summary_stack.append(text_batch)
                for i in range(len(summary_stack)):
                    tmp_text_batch = summary_stack[i]
                    mid = len(tmp_text_batch) // 2 # Split all the text batches in half
                    tmp_summary_stack.append(tmp_text_batch[mid:])
                    tmp_summary_stack.append(tmp_text_batch[:mid])
                summary_stack = tmp_summary_stack

        # If there is only 1 summary left we are done
        if len(summaries) == 1:
            return summaries[0]
        # Concatenate all the sumamries of batches and start the process again
        print(f"Pass {epoch} Complete - Now Summarizing {len(summaries)} Batches")
        epoch += 1
        summary_stack = [" ".join(summaries)]
        summaries = []

if __name__ == "__main__":
    text1 = "“Greetings parents and congratulations to Kenyon’s graduating class of 2005. There are these two young fish swimming along and they happen to meet an older fish swimming the other way, who nods at them and says “Morning, boys. How’s the water?” And the two young fish swim on for a bit, and then eventually one of them looks over at the other and goes “What the hell is water?” This is a standard requirement of US commencement speeches, the deployment of didactic little parable-ish stories. The story thing turns out to be one of the better, less bullshitty conventions of the genre, but if you’re worried that I plan to present myself here as the wise, older fish explaining what water is to you younger fish, please don’t be. I am not the wise old fish. The point of the fish story is merely that the most obvious, important realities are often the ones that are hardest to see and talk about. Stated as an English sentence, of course, this is just a banal platitude, but the fact is that in the day to day trenches of adult existence, banal platitudes can have a life or death importance, or so I wish to suggest to you on this dry and lovely morning. Of course, the main requirement of speeches like this is that I’m supposed to talk about your liberal arts education’s meaning, to try to explain why the degree you are about to receive has actual human value instead of just a material payoff. So let’s talk about the single most pervasive cliché in the commencement speech genre, which is that a liberal arts education is not so much about filling you up with knowledge as it is about “teaching you how to think.” If you’re like me as a student, you’ve never liked hearing this, and you tend to feel a bit insulted by the claim that you needed anybody to teach you how to think, since the fact that you even got admitted to a college this good seems like proof that you already know how to think. But I’m going to posit to you that the liberal arts cliché turns out not to be insulting at all, because the really significant education in thinking that we’re supposed to get in a place like this isn’t really about the capacity to think, but rather about the choice of what to think about. If your total freedom of choice regarding what to think about seems too obvious to waste time discussing, I’d ask you to think about fish and water, and to bracket for just a few minutes your skepticism about the value of the totally obvious. Here’s another didactic little story. There are these two guys sitting together in a bar in the remote Alaskan wilderness. One of the guys is religious, the other is an atheist, and the two are arguing about the existence of God with that special intensity that comes after about the fourth beer. And the atheist says: “Look, it’s not like I don’t have actual reasons for not believing in God. It’s not like I haven’t ever experimented with the whole God and prayer thing. Just last month I got caught away from the camp in that terrible blizzard, and I was totally lost and I couldn’t see a thing, and it was 50 below, and so I tried it: I fell to my knees in the snow and cried out ‘Oh, God, if there is a God, I’m lost in this blizzard, and I’m gonna die if you don’t help me.'” And now, in the bar, the religious guy looks at the atheist all puzzled. “Well then you must believe now,” he says, “After all, here you are, alive.” The atheist just rolls his eyes. “No, man, all that was was a couple Eskimos happened to come wandering by and showed me the way back to camp.” It’s easy to run this story through kind of a standard liberal arts analysis: the exact same experience can mean two totally different things to two different people, given those people’s two different belief templates and two different ways of constructing meaning from experience. Because we prize tolerance and diversity of belief, nowhere in our liberal arts analysis do we want to claim that one guy’s interpretation is true and the other guy’s is false or bad. Which is fine, except we also never end up talking about just where these individual templates and beliefs come from. Meaning, where they come from INSIDE the two guys. As if a person’s most basic orientation toward the world, and the meaning of his experience were somehow just hard-wired, like height or shoe-size; or automatically absorbed from the culture, like language. As if how we construct meaning were not actually a matter of personal, intentional choice. Plus, there’s the whole matter of arrogance. The nonreligious guy is so totally certain in his dismissal of the possibility that the passing Eskimos had anything to do with his prayer for help. True, there are plenty of religious people who seem arrogant and certain of their own interpretations, too. They’re probably even more repulsive than atheists, at least to most of us. But religious dogmatists’ problem is exactly the same as the story’s unbeliever: blind certainty, a close-mindedness that amounts to an imprisonment so total that the prisoner doesn’t even know he’s locked up. The point here is that I think this is one part of what teaching me how to think is really supposed to mean. To be just a little less arrogant. To have just a little critical awareness about myself and my certainties. Because a huge percentage of the stuff that I tend to be automatically certain of is, it turns out, totally wrong and deluded. I have learned this the hard way, as I predict you graduates will, too. Here is just one example of the total wrongness of something I tend to be automatically sure of: everything in my own immediate experience supports my deep belief that I am the absolute center of the universe; the realest, most vivid and important person in existence. We rarely think about this sort of natural, basic self-centredness because it’s so socially repulsive. But it’s pretty much the same for all of us. It is our default setting, hard-wired into our boards at birth. Think about it: there is no experience you have had that you are not the absolute center of. The world as you experience it is there in front of YOU or behind YOU, to the left or right of YOU, on YOUR TV or YOUR monitor. And so on. Other people’s thoughts and feelings have to be communicated to you somehow, but your own are so immediate, urgent, real. Please don’t worry that I’m getting ready to lecture you about compassion or other-directedness or all the so-called virtues. This is not a matter of virtue. It’s a matter of my choosing to do the work of somehow altering or getting free of my natural, hard-wired default setting which is to be deeply and literally self-centered and to see and interpret everything through this lens of self. People who can adjust their natural default setting this way are often described as being “well-adjusted”, which I suggest to you is not an accidental term. Given the triumphant academic setting here, an obvious question is how much of this work of adjusting our default setting involves actual knowledge or intellect. This question gets very tricky. Probably the most dangerous thing about an academic education–least in my own case–is that it enables my tendency to over-intellectualize stuff, to get lost in abstract argument inside my head, instead of simply paying attention to what is going on right in front of me, paying attention to what is going on inside me. As I’m sure you guys know by now, it is extremely difficult to stay alert and attentive, instead of getting hypnotized by the constant monologue inside your own head (may be happening right now). Twenty years after my own graduation, I have come gradually to understand that the liberal arts cliché about teaching you how to think is actually shorthand for a much deeper, more serious idea: learning how to think really means learning how to exercise some control over how and what you think. It means being conscious and aware enough to choose what you pay attention to and to choose how you construct meaning from experience. Because if you cannot exercise this kind of choice in adult life, you will be totally hosed. Think of the old cliché about “the mind being an excellent servant but a terrible master.” This, like many clichés, so lame and unexciting on the surface, actually expresses a great and terrible truth. It is not the least bit coincidental that adults who commit suicide with firearms almost always shoot themselves in: the head. They shoot the terrible master. And the truth is that most of these suicides are actually dead long before they pull the trigger. And I submit that this is what the real, no bullshit value of your liberal arts education is supposed to be about: how to keep from going through your comfortable, prosperous, respectable adult life dead, unconscious, a slave to your head and to your natural default setting of being uniquely, completely, imperially alone day in and day out. That may sound like hyperbole, or abstract nonsense. Let’s get concrete. The plain fact is that you graduating seniors do not yet have any clue what “day in day out” really means. There happen to be whole, large parts of adult American life that nobody talks about in commencement speeches. One such part involves boredom, routine and petty frustration. The parents and older folks here will know all too well what I’m talking about. By way of example, let’s say it’s an average adult day, and you get up in the morning, go to your challenging, white-collar, college-graduate job, and you work hard for eight or ten hours, and at the end of the day you’re tired and somewhat stressed and all you want is to go home and have a good supper and maybe unwind for an hour, and then hit the sack early because, of course, you have to get up the next day and do it all again. But then you remember there’s no food at home. You haven’t had time to shop this week because of your challenging job, and so now after work, you have to get in your car and drive to the supermarket. It’s the end of the work day and the traffic is apt to be very bad. So getting to the store takes way longer than it should, and when you finally get there, the supermarket is very crowded, because of course, it’s the time of day when all the other people with jobs also try to squeeze in some grocery shopping. And the store is hideously lit and infused with soul-killing muzak or corporate pop and it’s pretty much the last place you want to be but you can’t just get in and quickly out; you have to wander all over the huge, overlit store’s confusing aisles to find the stuff you want and you have to maneuver your junky cart through all these other tired, hurried people with carts (et cetera, et cetera, cutting stuff out because this is a long ceremony) and eventually you get all your supper supplies, except now it turns out there aren’t enough check-out lanes open even though it’s the end-of-the-day rush. So the checkout line is incredibly long, which is stupid and infuriating. But you can’t take your frustration out on the frantic lady working the register, who is overworked at a job whose daily tedium and meaninglessness surpasses the imagination of any of us here at a prestigious college. But anyway, you finally get to the checkout line’s front, and you pay for your food, and you get told to “Have a nice day” in a voice that is the absolute voice of death. Then you have to take your creepy, flimsy, plastic bags of groceries in your cart with the one crazy wheel that pulls maddeningly to the left, all the way out through the crowded, bumpy, littery parking lot, and then you have to drive all the way home through slow, heavy, SUV-intensive, rush-hour traffic, et cetera et cetera. Everyone here has done this, of course. But it hasn’t yet been part of you graduates’ actual life routine, day after week after month after year. But it will be. And many more dreary, annoying, seemingly meaningless routines besides. But that is not the point. The point is that petty, frustrating crap like this is exactly where the work of choosing is gonna come in. Because the traffic jams and crowded aisles and long checkout lines give me time to think, and if I don’t make a conscious decision about how to think and what to pay attention to, I’m gonna be pissed and miserable every time I have to shop. Because my natural default setting is the certainty that situations like this are really all about me. About MY hungriness and MY fatigue and MY desire to just get home, and it’s going to seem for all the world like everybody else is just in my way. And who are all these people in my way? And look at how repulsive most of them are, and how stupid and cow-like and dead-eyed and nonhuman they seem in the checkout line, or at how annoying and rude it is that people are talking loudly on cell phones in the middle of the line. And look at how deeply and personally unfair this is. Or, of course, if I’m in a more socially conscious liberal arts form of my default setting, I can spend time in the end-of-the-day traffic being disgusted about all the huge, stupid, lane-blocking SUV’s and Hummers and V-12 pickup trucks, burning their wasteful, selfish, 40-gallon tanks of gas, and I can dwell on the fact that the patriotic or religious bumper-stickers always seem to be on the biggest, most disgustingly selfish vehicles, driven by the ugliest [responding here to loud applause] — this is an example of how NOT to think, though — most disgustingly selfish vehicles, driven by the ugliest, most inconsiderate and aggressive drivers. And I can think about how our children’s children will despise us for wasting all the future’s fuel, and probably screwing up the climate, and how spoiled and stupid and selfish and disgusting we all are, and how modern consumer society just sucks, and so forth and so on. You get the idea."
    text2 = "by Beverly Daniel Tatum  The End of Policing by Alex Vitale  The New Jim Crow by Michelle Alexander  Guns of August by Barbara Tuchman  Discriminations and Disparities by Thomas Sowell  World as Will and Representation, Vol. 1 by Arthur Schopenhauer  Apocalypse Never by Michael Shellenberger  The Great Influenza by John Berry  The Pale Rider by Laura Spinney  Quack This Way by Bryan Garner and David Foster Wallace  Stamped From the Beginning by Ibram X Kendi  Intelligence: All That Matters by Stuart Ritchie  The Plague by Albert Camus  Science Fictions by Stuart Ritchie  Making Sense by Sam Harris  Letters from a Stoic by Seneca  Meditations on First Philosophy by Rene Descartes  World as Will and Representation, Vol 2.The Uninhabitable Earth by David Wallace-Wells  The Great Stagnation by Tyler Cowen  What Doesn’t Kill Us by Stephen Joseph, PhD  Existential Psychotherapy by Irvin Yalom  The Sports Gene by David Epstein  The Immortal Life of Henrietta Lacks by Rebecca Skloot  Elon Musk by Ashlee Vance  Why We’re Polarized by Ezra Klein  The Soulful Art of Persuasion by Jason Harris  Born Standing Up by Steve Martin  Zen and the Art of Writing by Ray Bradbury  The E-Myth Revisited by by Michael Gerber  Good to Great by Jim Collins  Candide by Voltaire  Steal Like an Artist by Austin Kleon  Lack and Transcendence by David Loy  Labyrinths by Jorge Luis Borges  Snow Crash by Neal Stephenson  Consider This by Chuck Palahniuk  Big Debt Crises by Ray Dalio  The Children of Time by Adrian Tchaikovsky  Drink?Check it out:  Recommended read: How to Be More Productive by Working Less  There’s been a burgeoning but fierce backlash against the so-called “attention economy,” which is the way in which modern tech companies have hacked their way into our brains in order to control our attention for nearly every waking moment of our lives:  …you know the story.  Compared to “normal” people (as if “normal” even exists), people with mental illnesses have more chronic physical health problems,5 have difficulty forming and maintaining relationships,6 earn less money, 7 and live shorter lives.  And for every quirky genius like Newton, who, in between re-inventing mathematics and formulating the fundamental laws of physics, probably had varied and interesting conversations with his mother’s sofa, you get people with mental health issues that do extraordinarily awful things as well — think The Unabomber, or crazed cult leaders, or school shooters, or even worse, a guy like Alex Jones:9  Mental health is a tricky subject though.As such, news. Hi there. "

    output = abstractive_summarize(text1, max_length=500)
    print(output)
