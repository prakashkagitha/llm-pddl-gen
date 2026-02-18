(define (problem coin_collector_numLocations7_numDistractorItems0_seed55)
  (:domain coin-collector)
  (:objects
    kitchen pantry corridor backyard bedroom living_room bathroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry north)
    (connected kitchen corridor east)
    (closed-door kitchen backyard west)
    (closed-door pantry kitchen south)
    (connected corridor kitchen west)
    (closed-door corridor bedroom south)
    (closed-door backyard kitchen east)
    (closed-door backyard living_room south)
    (closed-door bedroom corridor north)
    (closed-door living_room backyard north)
    (closed-door living_room bathroom south)
    (closed-door bathroom living_room north)
    (location coin living_room)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)