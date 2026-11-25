# Returns risk before dispatch

**To:** Ritu Deshpande, Head of D2C Operations, Kestrel Home
**From:** Dev Desai
**Date:** 25 November 2026
**Subject:** The model you asked for, the number it actually hits, and what I would do next week

**The decision.** You asked for a model that flags risky orders before dispatch so we can hold
them. I have built the model. I am asking you to change the action: flag the order and call the
customer, rather than hold it. Holding first costs you more than it saves, and I explain why below.

**The number.** Every order gets a risk score from 0 to 100 before it leaves the warehouse. I
tested the model on the last three months of orders it had never seen: 2,126 orders, 245 of which
came back. At the point where a call is worth more than it costs, it flags 673 of those orders. Of
the orders it flags, 25 in 100 actually come back. It catches 69 of every 100 returns and misses
31. So roughly a quarter of what it flags is a false alarm, and roughly a third of real returns
get through. Most of what it misses looks exactly like a good order: a first-time buyer, paid
upfront, no history with us. Nothing in our order data explains why that order came back.

**About 95%.** I cannot get there, and I do not think anyone can with this data. About 88 in 100
orders are kept, so a model that flags nothing at all is already 88.5% accurate, and the best any
setting reaches is 89.6%. To hit 95% you would have to make at most 106 mistakes across 2,126
orders when 245 orders genuinely came back. The information is not in the file. Accuracy is also
the wrong target: a model can be 89% accurate and useless, because it never flags anything and
never prevents a return. The number worth holding me to is returns prevented per rupee spent.

**The rupees.** A return costs ₹1,150, on Finance's policy figure rather than the ₹600 in the
original brief. A confirmation call costs ₹45, and your spring pilot suggests calls prevented about
35% of the returns they would have seen. That makes a call worth making once the risk of a return
is above about 11%, which is close to your average order. Over the three test months that works out
to about 59 returns prevented, ₹68,000 saved, ₹30,300 spent on calls, and **₹37,700 net**. That is
about ₹12,400 a month, or ₹1.5 lakh a year at your current volume. It is modest and it is real
money at almost no operating risk. If calls turn out to prevent only 20% of returns rather than
35%, it is still positive at about ₹2,800 a month.

**Why not hold.** A hold is not free. Your own policy says 12% of orders held beyond 24 hours get
cancelled, and every cancelled order costs the margin. It also lands hardest on Shield members:
they are 22% of orders but 36% of returns, and they are your highest-value customers. Excluding
them to protect the relationship gives up about ₹22,000 a quarter. Holding them is the worst of
both worlds. Calling them costs ₹45, keeps the relationship intact, and fixes the thing that is
usually wrong, which is a wrong model or a wrong address.

**Next week, three things.**

1. Approve calls, not holds, for the top band of flagged orders from Monday. Keep a hold only for
   the small number of flagged orders where the call cannot reach the customer.
2. Run it for four weeks and count returns on called orders against a similar group we did not
   call. That settles whether the 35% figure holds in your operation.
3. Judge it on returns prevented, calls made and cancellations caused. Not on accuracy.

One thing would sharpen the hold case: Finance's gross margin per order. It is not in the pack, and
it is the only input needed to price a hold properly rather than assume it.

Dev Desai
