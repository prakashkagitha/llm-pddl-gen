(define (problem coin_collector_numLocations5_numDistractorItems0_seed48)
  (:domain coin-collector)
  (:objects
    kitchen pantry backyard corridor bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry north)
    (closed-door kitchen backyard south)
    (closed-door pantry kitchen south)
    (closed-door backyard kitchen north)
    (closed-door backyard corridor east)
    (closed-door corridor backyard west)
    (closed-door corridor bedroom south)
    (closed-door bedroom corridor north)
    (location coin corridor)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)