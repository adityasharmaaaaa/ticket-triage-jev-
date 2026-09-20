from typing import NamedTuple


class Ticket(NamedTuple):
    id: str
    message: str


# A mix on purpose: clear cases, murky ones, risky ones, and noise.
TICKETS = [
    Ticket("T-101", "I was charged twice for my March invoice. Please refund the extra charge."),
    Ticket("T-102", "The dashboard has returned a 500 error since this morning and my whole team is blocked."),
    Ticket("T-103", "How do I change the email address on my account?"),
    Ticket("T-104", "Your last update broke my export, and honestly I'm done. Cancel my plan and refund this quarter or I'm calling my lawyer."),
    Ticket("T-105", "Thanks so much, that fixed it! Great support as always."),
    Ticket("T-106", "Somebody logged into our admin account from a country we don't operate in. I think our API keys may be exposed."),
    Ticket("T-107", "Can I get a quote for 200 seats? Also the SSO login has been flaky for a few days."),
    Ticket("T-108", "BUY CHEAP WATCHES NOW!!! limited offer click here"),
]
